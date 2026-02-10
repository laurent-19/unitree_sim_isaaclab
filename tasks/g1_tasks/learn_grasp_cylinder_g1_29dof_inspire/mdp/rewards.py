# Copyright (c) 2025, Unitree Robotics Co., Ltd. All Rights Reserved.
# License: Apache License, Version 2.0

"""
Reward functions for Learn-to-Grasp task.
Rewards finger contact, grasp stability, lifting, and successful placement.
"""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

from isaaclab.managers import SceneEntityCfg

# Import phase constants - support both package and direct import modes
try:
    from .events import PHASE_APPROACH, PHASE_GRASP, PHASE_LIFT, PHASE_MOVE_TO_BIN, PHASE_RELEASE
except ImportError:
    import sys
    _events = sys.modules.get("learn_grasp_mdp_events")
    if _events:
        PHASE_APPROACH = _events.PHASE_APPROACH
        PHASE_GRASP = _events.PHASE_GRASP
        PHASE_LIFT = _events.PHASE_LIFT
        PHASE_MOVE_TO_BIN = _events.PHASE_MOVE_TO_BIN
        PHASE_RELEASE = _events.PHASE_RELEASE
    else:
        # Fallback constants
        PHASE_APPROACH, PHASE_GRASP, PHASE_LIFT, PHASE_MOVE_TO_BIN, PHASE_RELEASE = 0, 1, 2, 3, 4


def _get_observations_module():
    """Get observations module, supporting both package and direct import."""
    try:
        from . import observations
        return observations
    except ImportError:
        import sys
        return sys.modules.get("learn_grasp_mdp_observations")


def finger_cylinder_contact_reward(
    env: ManagerBasedRLEnv,
    min_force: float = 0.1,
    max_force: float = 10.0,
) -> torch.Tensor:
    """Reward for making contact with the cylinder.

    Higher reward for more fingers in contact with appropriate force.
    Active during GRASP, LIFT, and MOVE phases.

    Args:
        env: Environment instance
        min_force: Minimum force to count as contact
        max_force: Force for maximum reward per finger

    Returns:
        torch.Tensor: [num_envs] reward values (0-1 normalized)
    """
    obs_mod = _get_observations_module()
    get_right_hand_contact_forces = obs_mod.get_right_hand_contact_forces

    forces = get_right_hand_contact_forces(env)  # [num_envs, 6]

    # Only reward during grasp-related phases
    if hasattr(env, "_task_phase"):
        active_phases = (env._task_phase >= PHASE_GRASP) & (env._task_phase <= PHASE_MOVE_TO_BIN)
    else:
        active_phases = torch.ones(env.num_envs, dtype=torch.bool, device=env.device)

    # Clamp forces and normalize
    clamped_forces = forces.clamp(min=0.0, max=max_force)

    # Count fingers with meaningful contact
    fingers_in_contact = (clamped_forces > min_force).float()

    # Reward scales with number of fingers and force magnitude
    contact_score = fingers_in_contact.sum(dim=1) / 6.0  # Normalize by max fingers
    force_score = (clamped_forces / max_force).mean(dim=1)  # Normalized avg force

    reward = (contact_score + force_score) * 0.5  # Average of both scores

    # Zero out reward for non-active phases
    reward = reward * active_phases.float()

    return reward


def cylinder_velocity_penalty(
    env: ManagerBasedRLEnv,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    max_velocity: float = 0.5,
) -> torch.Tensor:
    """Penalize fast cylinder movement (indicates unstable grasp).

    Active during GRASP, LIFT, and MOVE phases.

    Args:
        env: Environment instance
        object_cfg: Scene entity config for cylinder
        max_velocity: Velocity above which penalty is maximum

    Returns:
        torch.Tensor: [num_envs] penalty values (0-1, higher = worse)
    """
    obj = env.scene[object_cfg.name]
    vel = obj.data.root_lin_vel_w  # [num_envs, 3]

    vel_magnitude = torch.norm(vel, dim=-1)

    # Normalize to 0-1 range
    penalty = (vel_magnitude / max_velocity).clamp(0.0, 1.0)

    # Only penalize during grasp phases
    if hasattr(env, "_task_phase"):
        active_phases = (env._task_phase >= PHASE_GRASP) & (env._task_phase <= PHASE_MOVE_TO_BIN)
        penalty = penalty * active_phases.float()

    return penalty


def cylinder_lift_reward(
    env: ManagerBasedRLEnv,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    initial_height: float = 0.84,
    target_height: float = 1.10,
) -> torch.Tensor:
    """Reward for lifting the cylinder.

    Active primarily during LIFT phase.

    Args:
        env: Environment instance
        object_cfg: Scene entity config for cylinder
        initial_height: Starting Z position
        target_height: Target Z position for full reward

    Returns:
        torch.Tensor: [num_envs] reward (0-1 based on lift progress)
    """
    obj = env.scene[object_cfg.name]
    height = obj.data.root_pos_w[:, 2]

    # Calculate lift progress
    lift_progress = (height - initial_height) / (target_height - initial_height)
    lift_progress = lift_progress.clamp(0.0, 1.0)

    # Only reward during lift phase
    if hasattr(env, "_task_phase"):
        in_lift = env._task_phase == PHASE_LIFT
        lift_progress = lift_progress * in_lift.float()

    return lift_progress


def cylinder_in_bin_reward(
    env: ManagerBasedRLEnv,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    bin_center_x: float = 0.3,
    bin_center_y: float = -0.2,
    bin_height: float = 0.85,
    radius: float = 0.15,
) -> torch.Tensor:
    """Reward for placing cylinder in target bin area.

    Active during RELEASE phase.

    Args:
        env: Environment instance
        object_cfg: Scene entity config for cylinder
        bin_center_x: X coordinate of bin center
        bin_center_y: Y coordinate of bin center
        bin_height: Expected Z height in bin
        radius: Radius for "in bin" detection

    Returns:
        torch.Tensor: [num_envs] reward (1.0 if in bin, 0 otherwise)
    """
    obj = env.scene[object_cfg.name]
    pos = obj.data.root_pos_w  # [num_envs, 3]

    # Distance from bin center (XY plane)
    dist_xy = torch.sqrt(
        (pos[:, 0] - bin_center_x) ** 2 + (pos[:, 1] - bin_center_y) ** 2
    )

    # Check if cylinder is in bin area
    in_bin_xy = dist_xy < radius
    in_bin_z = pos[:, 2] < bin_height + 0.1  # Within reasonable height

    in_bin = in_bin_xy & in_bin_z

    return in_bin.float()


def success_bonus(
    env: ManagerBasedRLEnv,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    bin_center_x: float = 0.3,
    bin_center_y: float = -0.2,
    bin_height: float = 0.85,
    radius: float = 0.15,
) -> torch.Tensor:
    """Large one-time bonus for successful task completion.

    Given when cylinder is in bin and fingers are open.

    Args:
        env: Environment instance
        object_cfg: Scene entity config for cylinder
        bin_center/height/radius: Bin location parameters

    Returns:
        torch.Tensor: [num_envs] bonus (1.0 for success, 0 otherwise)
    """
    obs_mod = _get_observations_module()
    get_right_hand_joint_pos = obs_mod.get_right_hand_joint_pos

    # Check cylinder in bin
    in_bin = cylinder_in_bin_reward(
        env, object_cfg, bin_center_x, bin_center_y, bin_height, radius
    )

    # Check fingers are mostly open (released)
    finger_pos = get_right_hand_joint_pos(env)  # [num_envs, 6]
    fingers_open = (finger_pos.abs() < 0.3).all(dim=1)  # All fingers near zero

    # Only in release phase
    if hasattr(env, "_task_phase"):
        in_release = env._task_phase == PHASE_RELEASE
    else:
        in_release = torch.ones(env.num_envs, dtype=torch.bool, device=env.device)

    success = in_bin.bool() & fingers_open & in_release

    return success.float()


def excessive_force_penalty(
    env: ManagerBasedRLEnv,
    force_threshold: float = 20.0,
) -> torch.Tensor:
    """Penalize excessive finger forces (crushing).

    Args:
        env: Environment instance
        force_threshold: Force above which penalty applies

    Returns:
        torch.Tensor: [num_envs] penalty (0-1)
    """
    obs_mod = _get_observations_module()
    get_right_hand_contact_forces = obs_mod.get_right_hand_contact_forces

    forces = get_right_hand_contact_forces(env)  # [num_envs, 6]

    # Max force across fingers
    max_force = forces.max(dim=1).values

    # Penalty for forces above threshold
    excess = (max_force - force_threshold).clamp(min=0.0)
    penalty = (excess / force_threshold).clamp(0.0, 1.0)

    return penalty


def approach_phase_reward(
    env: ManagerBasedRLEnv,
) -> torch.Tensor:
    """Small reward during approach phase to encourage waiting.

    Policy receives positive reward for keeping fingers open during approach.

    Returns:
        torch.Tensor: [num_envs] reward
    """
    obs_mod = _get_observations_module()
    get_right_hand_joint_pos = obs_mod.get_right_hand_joint_pos

    # Check if in approach phase
    if hasattr(env, "_task_phase"):
        in_approach = env._task_phase == PHASE_APPROACH
    else:
        return torch.zeros(env.num_envs, device=env.device)

    # Reward for keeping fingers open
    finger_pos = get_right_hand_joint_pos(env)
    fingers_open = (finger_pos.abs() < 0.2).all(dim=1)

    reward = in_approach.float() * fingers_open.float() * 0.1

    return reward


def grasp_phase_progress_reward(
    env: ManagerBasedRLEnv,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Reward for making grasp progress during grasp phase.

    Combines finger contact and cylinder stability.

    Returns:
        torch.Tensor: [num_envs] reward
    """
    contact_rew = finger_cylinder_contact_reward(env, min_force=0.2, max_force=5.0)
    stability_pen = cylinder_velocity_penalty(env, object_cfg, max_velocity=0.3)

    # Only during grasp phase
    if hasattr(env, "_task_phase"):
        in_grasp = env._task_phase == PHASE_GRASP
        return (contact_rew - 0.5 * stability_pen) * in_grasp.float()

    return contact_rew - 0.5 * stability_pen


__all__ = [
    "finger_cylinder_contact_reward",
    "cylinder_velocity_penalty",
    "cylinder_lift_reward",
    "cylinder_in_bin_reward",
    "success_bonus",
    "excessive_force_penalty",
    "approach_phase_reward",
    "grasp_phase_progress_reward",
]
