# Copyright (c) 2025, Unitree Robotics Co., Ltd. All Rights Reserved.
# License: Apache License, Version 2.0

"""
Observation functions for Learn-to-Grasp task.
Provides finger positions, contact forces, cylinder pose, and task phase.
"""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

from isaaclab.managers import SceneEntityCfg


# Right hand finger joint names (Inspire hand)
RIGHT_HAND_JOINT_NAMES = [
    "R_index_proximal_joint",
    "R_middle_proximal_joint",
    "R_ring_proximal_joint",
    "R_pinky_proximal_joint",
    "R_thumb_proximal_pitch_joint",
    "R_thumb_proximal_yaw_joint",
]

# Right arm joint names for IK control
RIGHT_ARM_JOINT_NAMES = [
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
]

# Cache for joint indices
_obs_cache = {
    "device": None,
    "right_hand_idx": None,
    "right_arm_idx": None,
}


def _get_joint_indices(env: ManagerBasedRLEnv, joint_names: list, cache_key: str) -> torch.Tensor:
    """Get cached joint indices for given joint names."""
    global _obs_cache
    device = env.device

    if _obs_cache["device"] != device or _obs_cache[cache_key] is None:
        robot = env.scene["robot"]
        all_joint_names = robot.data.joint_names
        indices = []
        for name in joint_names:
            try:
                idx = all_joint_names.index(name)
                indices.append(idx)
            except ValueError:
                print(f"[observations] Warning: Joint '{name}' not found in robot")
        _obs_cache[cache_key] = torch.tensor(indices, dtype=torch.long, device=device)
        _obs_cache["device"] = device

    return _obs_cache[cache_key]


def get_right_hand_joint_pos(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Get right hand finger joint positions.

    Returns:
        torch.Tensor: [num_envs, 6] - finger joint positions in radians
    """
    robot = env.scene["robot"]
    joint_pos = robot.data.joint_pos

    idx = _get_joint_indices(env, RIGHT_HAND_JOINT_NAMES, "right_hand_idx")

    if idx.numel() == 0:
        return torch.zeros(env.num_envs, 6, device=env.device)

    # Gather positions for right hand joints
    batch_idx = idx.unsqueeze(0).expand(env.num_envs, -1)
    return torch.gather(joint_pos, 1, batch_idx)


def get_right_hand_joint_vel(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Get right hand finger joint velocities.

    Returns:
        torch.Tensor: [num_envs, 6] - finger joint velocities
    """
    robot = env.scene["robot"]
    joint_vel = robot.data.joint_vel

    idx = _get_joint_indices(env, RIGHT_HAND_JOINT_NAMES, "right_hand_idx")

    if idx.numel() == 0:
        return torch.zeros(env.num_envs, 6, device=env.device)

    batch_idx = idx.unsqueeze(0).expand(env.num_envs, -1)
    return torch.gather(joint_vel, 1, batch_idx)


def get_right_hand_contact_forces(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Get contact force magnitudes for right hand fingers.

    Maps to FORCE_ACT register (1582) on real Inspire hand.

    Returns:
        torch.Tensor: [num_envs, 6] - force magnitudes for each finger
                      [index, middle, ring, pinky, thumb_pitch, thumb_yaw/palm]
    """
    device = env.device
    batch = env.num_envs

    # Initialize output
    finger_forces = torch.zeros(batch, 6, device=device)

    try:
        # Check if contact sensor exists
        if not hasattr(env.scene, "contact_forces"):
            return finger_forces

        contact_sensor = env.scene["contact_forces"]
        if contact_sensor is None:
            return finger_forces

        # Get net contact forces - shape: [batch, num_bodies, 3]
        net_forces = contact_sensor.data.net_forces_w
        if net_forces is None:
            return finger_forces

        # Compute force magnitudes
        force_magnitudes = torch.norm(net_forces, dim=-1)  # [batch, num_bodies]

        # Get body names from robot
        robot = env.scene["robot"]
        body_names = robot.data.body_names if hasattr(robot.data, 'body_names') else []

        # Right hand finger body names (tips/intermediate links)
        right_finger_bodies = [
            "R_index_intermediate",
            "R_middle_intermediate",
            "R_ring_intermediate",
            "R_pinky_intermediate",
            "R_thumb_distal",
            "right_wrist_roll_link",  # palm area
        ]

        # Map body names to force indices
        for finger_idx, body_name in enumerate(right_finger_bodies):
            for body_idx, name in enumerate(body_names):
                if body_name.lower() in name.lower():
                    if body_idx < force_magnitudes.shape[1]:
                        finger_forces[:, finger_idx] = force_magnitudes[:, body_idx]
                    break

        return finger_forces

    except Exception as e:
        return finger_forces


def get_cylinder_position(
    env: ManagerBasedRLEnv,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Get cylinder world position.

    Returns:
        torch.Tensor: [num_envs, 3] - cylinder XYZ position
    """
    obj = env.scene[object_cfg.name]
    return obj.data.root_pos_w.clone()


def get_cylinder_height(
    env: ManagerBasedRLEnv,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Get cylinder height (Z position).

    Returns:
        torch.Tensor: [num_envs, 1] - cylinder Z coordinate
    """
    obj = env.scene[object_cfg.name]
    return obj.data.root_pos_w[:, 2:3].clone()


def get_cylinder_velocity(
    env: ManagerBasedRLEnv,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Get cylinder linear velocity.

    Returns:
        torch.Tensor: [num_envs, 3] - cylinder velocity
    """
    obj = env.scene[object_cfg.name]
    return obj.data.root_lin_vel_w.clone()


def get_cylinder_relative_to_palm(
    env: ManagerBasedRLEnv,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    palm_body_name: str = "right_wrist_roll_link",
) -> torch.Tensor:
    """Get cylinder position relative to right palm.

    Returns:
        torch.Tensor: [num_envs, 3] - cylinder position in palm frame
    """
    robot = env.scene["robot"]
    obj = env.scene[object_cfg.name]

    # Get palm world position
    body_names = robot.data.body_names
    try:
        palm_idx = body_names.index(palm_body_name)
    except ValueError:
        # Fallback: return world position offset
        return obj.data.root_pos_w.clone()

    palm_pos_w = robot.data.body_pos_w[:, palm_idx, :]  # [num_envs, 3]
    palm_quat_w = robot.data.body_quat_w[:, palm_idx, :]  # [num_envs, 4]

    # Cylinder world position
    cyl_pos_w = obj.data.root_pos_w  # [num_envs, 3]

    # Vector from palm to cylinder in world frame
    delta_w = cyl_pos_w - palm_pos_w

    # Transform to palm frame (simple version - just translate, ignore rotation for now)
    # For full transform, we'd need to apply inverse rotation
    # This simplified version gives relative position which is often sufficient
    return delta_w


def get_task_phase(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Get current task phase as one-hot vector.

    Phases: 0=approach, 1=grasp, 2=lift, 3=move_to_bin, 4=release

    Returns:
        torch.Tensor: [num_envs, 5] - one-hot phase encoding
    """
    device = env.device
    batch = env.num_envs

    # Get phase from environment buffer (set by events)
    if not hasattr(env, "_task_phase"):
        env._task_phase = torch.zeros(batch, dtype=torch.long, device=device)

    phase = env._task_phase.clamp(0, 4)

    # Convert to one-hot
    one_hot = torch.zeros(batch, 5, device=device)
    one_hot.scatter_(1, phase.unsqueeze(1), 1.0)

    return one_hot


def get_task_phase_scalar(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Get current task phase as normalized scalar.

    Returns:
        torch.Tensor: [num_envs, 1] - phase / 4.0 (normalized 0-1)
    """
    device = env.device
    batch = env.num_envs

    if not hasattr(env, "_task_phase"):
        env._task_phase = torch.zeros(batch, dtype=torch.long, device=device)

    return (env._task_phase.float() / 4.0).unsqueeze(1)


def get_right_palm_position(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Get right palm world position.

    Returns:
        torch.Tensor: [num_envs, 3] - palm XYZ position
    """
    robot = env.scene["robot"]
    body_names = robot.data.body_names

    try:
        palm_idx = body_names.index("right_wrist_roll_link")
    except ValueError:
        return torch.zeros(env.num_envs, 3, device=env.device)

    return robot.data.body_pos_w[:, palm_idx, :].clone()


__all__ = [
    "get_right_hand_joint_pos",
    "get_right_hand_joint_vel",
    "get_right_hand_contact_forces",
    "get_cylinder_position",
    "get_cylinder_height",
    "get_cylinder_velocity",
    "get_cylinder_relative_to_palm",
    "get_task_phase",
    "get_task_phase_scalar",
    "get_right_palm_position",
    "RIGHT_HAND_JOINT_NAMES",
    "RIGHT_ARM_JOINT_NAMES",
]
