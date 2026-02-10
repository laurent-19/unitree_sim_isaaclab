# Copyright (c) 2025, Unitree Robotics Co., Ltd. All Rights Reserved.
# License: Apache License, Version 2.0

"""
Event functions for Learn-to-Grasp task.
Handles scripted arm movement through task phases.
"""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING, Dict, Tuple, Optional

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

from isaaclab.managers import SceneEntityCfg

# Phase definitions
PHASE_APPROACH = 0
PHASE_GRASP = 1
PHASE_LIFT = 2
PHASE_MOVE_TO_BIN = 3
PHASE_RELEASE = 4

# Right arm joint names
RIGHT_ARM_JOINTS = [
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
]

# Pre-computed arm configurations for each phase (in radians)
# These are target joint positions that place the hand at desired locations
# Tuned for the G1 robot with cylinder at (-0.35, 0.40, 0.84)
PHASE_ARM_TARGETS = {
    PHASE_APPROACH: {
        # Hand approaching cylinder from above
        "right_shoulder_pitch_joint": 0.3,
        "right_shoulder_roll_joint": -0.4,
        "right_shoulder_yaw_joint": 0.2,
        "right_elbow_joint": -0.8,
        "right_wrist_roll_joint": 0.0,
        "right_wrist_pitch_joint": 0.3,
        "right_wrist_yaw_joint": 0.0,
    },
    PHASE_GRASP: {
        # Hand positioned around cylinder for grasping
        "right_shoulder_pitch_joint": 0.4,
        "right_shoulder_roll_joint": -0.35,
        "right_shoulder_yaw_joint": 0.25,
        "right_elbow_joint": -0.9,
        "right_wrist_roll_joint": 0.0,
        "right_wrist_pitch_joint": 0.4,
        "right_wrist_yaw_joint": 0.0,
    },
    PHASE_LIFT: {
        # Hand lifting cylinder up
        "right_shoulder_pitch_joint": 0.2,
        "right_shoulder_roll_joint": -0.3,
        "right_shoulder_yaw_joint": 0.2,
        "right_elbow_joint": -0.6,
        "right_wrist_roll_joint": 0.0,
        "right_wrist_pitch_joint": 0.3,
        "right_wrist_yaw_joint": 0.0,
    },
    PHASE_MOVE_TO_BIN: {
        # Hand moving cylinder toward bin location
        "right_shoulder_pitch_joint": 0.1,
        "right_shoulder_roll_joint": -0.5,
        "right_shoulder_yaw_joint": -0.3,
        "right_elbow_joint": -0.5,
        "right_wrist_roll_joint": 0.0,
        "right_wrist_pitch_joint": 0.2,
        "right_wrist_yaw_joint": 0.0,
    },
    PHASE_RELEASE: {
        # Hand at bin, ready to release
        "right_shoulder_pitch_joint": 0.15,
        "right_shoulder_roll_joint": -0.55,
        "right_shoulder_yaw_joint": -0.35,
        "right_elbow_joint": -0.55,
        "right_wrist_roll_joint": 0.0,
        "right_wrist_pitch_joint": 0.25,
        "right_wrist_yaw_joint": 0.0,
    },
}

# Duration (in steps) for each phase
PHASE_DURATIONS = {
    PHASE_APPROACH: 50,
    PHASE_GRASP: 60,
    PHASE_LIFT: 50,
    PHASE_MOVE_TO_BIN: 60,
    PHASE_RELEASE: 30,
}

# Cache for arm joint indices
_event_cache = {
    "device": None,
    "arm_idx": None,
}


def _get_arm_indices(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Get cached arm joint indices."""
    global _event_cache
    device = env.device

    if _event_cache["device"] != device or _event_cache["arm_idx"] is None:
        robot = env.scene["robot"]
        all_joint_names = robot.data.joint_names
        indices = []
        for name in RIGHT_ARM_JOINTS:
            try:
                idx = all_joint_names.index(name)
                indices.append(idx)
            except ValueError:
                print(f"[events] Warning: Arm joint '{name}' not found")
        _event_cache["arm_idx"] = torch.tensor(indices, dtype=torch.long, device=device)
        _event_cache["device"] = device

    return _event_cache["arm_idx"]


def reset_task_phase(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
) -> None:
    """Reset task phase counter on episode reset.

    Args:
        env: Environment instance
        env_ids: Environment indices to reset
    """
    if not hasattr(env, "_task_phase"):
        env._task_phase = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)

    if not hasattr(env, "_phase_step"):
        env._phase_step = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)

    env._task_phase[env_ids] = PHASE_APPROACH
    env._phase_step[env_ids] = 0


def reset_grasp_latch(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
) -> None:
    """Reset grasp success latch on episode reset."""
    if not hasattr(env, "_grasp_success"):
        env._grasp_success = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

    env._grasp_success[env_ids] = False


def scripted_arm_control(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    interpolation_speed: float = 0.1,
) -> None:
    """Control arm joints based on current task phase.

    This function is called every step to update arm joint targets.
    The arm smoothly interpolates toward the target configuration
    for the current phase.

    Args:
        env: Environment instance
        env_ids: Environment indices to update
        interpolation_speed: How fast to interpolate (0-1, higher = faster)
    """
    if len(env_ids) == 0:
        return

    # Initialize buffers if needed
    if not hasattr(env, "_task_phase"):
        env._task_phase = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
    if not hasattr(env, "_phase_step"):
        env._phase_step = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)

    robot = env.scene["robot"]
    arm_idx = _get_arm_indices(env)

    if arm_idx.numel() == 0:
        return

    # Get current arm positions
    current_pos = robot.data.joint_pos[:, arm_idx]  # [num_envs, 7]

    # Build target tensor for each environment based on phase
    target_pos = torch.zeros_like(current_pos)

    for phase_id, targets in PHASE_ARM_TARGETS.items():
        # Find environments in this phase
        mask = env._task_phase == phase_id

        if mask.any():
            # Build target tensor for this phase
            phase_target = torch.tensor(
                [targets[name] for name in RIGHT_ARM_JOINTS],
                device=env.device,
                dtype=current_pos.dtype,
            )
            target_pos[mask] = phase_target

    # Smooth interpolation toward target
    new_pos = current_pos + interpolation_speed * (target_pos - current_pos)

    # Apply to robot
    robot.data.joint_pos[:, arm_idx] = new_pos

    # Also set as target for position control
    if hasattr(robot.data, 'joint_pos_target'):
        robot.data.joint_pos_target[:, arm_idx] = new_pos

    # Increment phase step counter
    env._phase_step += 1

    # Check for phase transitions
    _update_phase_transitions(env)


def _update_phase_transitions(env: ManagerBasedRLEnv) -> None:
    """Update task phases based on step counts and conditions."""
    device = env.device

    for phase_id, duration in PHASE_DURATIONS.items():
        # Find envs in this phase that exceeded duration
        in_phase = env._task_phase == phase_id
        exceeded_duration = env._phase_step >= duration

        should_transition = in_phase & exceeded_duration

        if should_transition.any():
            # Transition to next phase
            next_phase = min(phase_id + 1, PHASE_RELEASE)
            env._task_phase[should_transition] = next_phase
            env._phase_step[should_transition] = 0


def advance_phase_on_condition(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    from_phase: int,
    to_phase: int,
    condition_fn: callable,
) -> None:
    """Advance phase when a condition is met.

    Can be used for early phase transitions, e.g., when grasp is detected.

    Args:
        env: Environment instance
        env_ids: Environment indices to check
        from_phase: Current phase that should transition
        to_phase: Target phase
        condition_fn: Function(env, env_ids) -> bool tensor
    """
    if not hasattr(env, "_task_phase"):
        return

    in_phase = env._task_phase[env_ids] == from_phase
    if not in_phase.any():
        return

    condition_met = condition_fn(env, env_ids)
    should_transition = in_phase & condition_met

    if should_transition.any():
        transition_ids = env_ids[should_transition]
        env._task_phase[transition_ids] = to_phase
        env._phase_step[transition_ids] = 0


def check_grasp_success(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    min_contact_force: float = 0.5,
    min_fingers_contact: int = 2,
) -> torch.Tensor:
    """Check if grasp is successful based on contact forces.

    Args:
        env: Environment instance
        env_ids: Environment indices to check
        min_contact_force: Minimum force threshold per finger
        min_fingers_contact: Minimum number of fingers with contact

    Returns:
        torch.Tensor: Boolean tensor indicating grasp success
    """
    # Import observation function - support both package and direct import
    try:
        from .observations import get_right_hand_contact_forces
    except ImportError:
        import sys
        obs_mod = sys.modules.get("learn_grasp_mdp_observations")
        get_right_hand_contact_forces = obs_mod.get_right_hand_contact_forces

    forces = get_right_hand_contact_forces(env)  # [num_envs, 6]
    forces_subset = forces[env_ids]

    # Count fingers with sufficient contact
    fingers_in_contact = (forces_subset > min_contact_force).sum(dim=1)

    return fingers_in_contact >= min_fingers_contact


__all__ = [
    "reset_task_phase",
    "reset_grasp_latch",
    "scripted_arm_control",
    "advance_phase_on_condition",
    "check_grasp_success",
    "PHASE_APPROACH",
    "PHASE_GRASP",
    "PHASE_LIFT",
    "PHASE_MOVE_TO_BIN",
    "PHASE_RELEASE",
    "PHASE_DURATIONS",
    "RIGHT_ARM_JOINTS",
]
