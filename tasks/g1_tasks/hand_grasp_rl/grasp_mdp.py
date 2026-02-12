# Copyright (c) 2025, Unitree Robotics Co., Ltd. All Rights Reserved.
# License: Apache License, Version 2.0

"""MDP functions for hand grasp RL task - Stage 2 of two-stage pipeline.

Contains observation, reward, and termination functions for training
a hand-only grasping policy.
"""

from __future__ import annotations
from typing import TYPE_CHECKING

import torch

from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import subtract_frame_transforms

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------

def _to_1d_list(x) -> list:
    """Convert SceneEntityCfg fields to list."""
    if x is None:
        return []
    if isinstance(x, (list, tuple)):
        return list(x)
    return [x]


def _resolve_single_body_id(asset: Articulation, asset_cfg: SceneEntityCfg) -> int:
    """Resolve a single body index from SceneEntityCfg."""
    body_ids = getattr(asset_cfg, "body_ids", None)

    if body_ids is not None and not isinstance(body_ids, slice):
        if isinstance(body_ids, torch.Tensor):
            return int(body_ids.flatten()[0].item())
        return int(list(body_ids)[0])

    body_names = _to_1d_list(getattr(asset_cfg, "body_names", None))
    if len(body_names) != 1:
        raise ValueError(f"Expected exactly one body. Got body_names={body_names}")

    name0 = body_names[0]

    if hasattr(asset, "find_bodies"):
        res = asset.find_bodies([name0])
        ids = res[0] if isinstance(res, tuple) else res
        return int(ids[0])

    if hasattr(asset, "data") and hasattr(asset.data, "body_names"):
        return int(list(asset.data.body_names).index(name0))

    raise AttributeError("Cannot resolve body id")


# Cached joint indices for hand
_hand_joint_cache = {
    "initialized": False,
    "indices": None,
}


def _get_hand_joint_indices(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Get joint indices for right hand by name lookup (cached)."""
    global _hand_joint_cache

    if not _hand_joint_cache["initialized"]:
        robot = env.scene["robot"]
        joint_names = robot.data.joint_names

        # Right hand proximal joint names
        target_names = [
            "R_index_proximal_joint",
            "R_middle_proximal_joint",
            "R_ring_proximal_joint",
            "R_pinky_proximal_joint",
            "R_thumb_proximal_pitch_joint",
            "R_thumb_proximal_yaw_joint",
        ]

        indices = []
        for name in target_names:
            try:
                idx = joint_names.index(name)
                indices.append(idx)
            except ValueError:
                print(f"[grasp_mdp] Warning: Joint '{name}' not found in robot")
                indices.append(0)

        _hand_joint_cache["indices"] = torch.tensor(indices, dtype=torch.long, device=env.device)
        _hand_joint_cache["initialized"] = True

    return _hand_joint_cache["indices"]


# -----------------------------------------------------------------------------
# Observations
# -----------------------------------------------------------------------------

def hand_joint_pos(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Get hand joint positions (6 DOF).

    Returns:
        Tensor of shape (num_envs, 6)
    """
    robot = env.scene["robot"]
    joint_pos = robot.data.joint_pos
    batch = joint_pos.shape[0]

    idx_t = _get_hand_joint_indices(env)
    idx_batch = idx_t.unsqueeze(0).expand(batch, -1)

    return torch.gather(joint_pos, 1, idx_batch)


def hand_joint_vel(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Get hand joint velocities (6 DOF).

    Returns:
        Tensor of shape (num_envs, 6)
    """
    robot = env.scene["robot"]
    joint_vel = robot.data.joint_vel
    batch = joint_vel.shape[0]

    idx_t = _get_hand_joint_indices(env)
    idx_batch = idx_t.unsqueeze(0).expand(batch, -1)

    return torch.gather(joint_vel, 1, idx_batch)


def contact_forces(env: ManagerBasedRLEnv, sensor_name: str = "fingertip_contacts") -> torch.Tensor:
    """Get contact forces from fingertip sensors.

    Returns:
        Tensor of shape (num_envs, num_bodies * 3) - flattened force vectors
    """
    if sensor_name not in env.scene.sensors:
        # Return zeros if sensor not configured
        return torch.zeros(env.num_envs, 15, device=env.device)

    sensor = env.scene[sensor_name]
    forces = sensor.data.net_forces_w  # (num_envs, num_bodies, 3)
    # Flatten to (num_envs, num_bodies * 3)
    return forces.view(env.num_envs, -1)


def object_pose_relative_to_hand(
    env: ManagerBasedRLEnv,
    object_cfg: SceneEntityCfg = SceneEntityCfg("grasp_cylinder"),
    ee_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=["R_thumb_proximal"]),
) -> torch.Tensor:
    """Get object pose relative to hand/end-effector.

    Returns:
        Tensor of shape (num_envs, 7) - position (3) + quaternion (4)
    """
    obj: RigidObject = env.scene[object_cfg.name]
    robot: Articulation = env.scene[ee_cfg.name]

    ee_body_id = _resolve_single_body_id(robot, ee_cfg)

    # Object pose in world
    obj_pos_w = obj.data.root_pos_w
    obj_quat_w = obj.data.root_quat_w

    # EE pose in world
    ee_pos_w = robot.data.body_pos_w[:, ee_body_id]
    ee_quat_w = robot.data.body_quat_w[:, ee_body_id]

    # Object position relative to EE
    rel_pos, rel_quat = subtract_frame_transforms(ee_pos_w, ee_quat_w, obj_pos_w, obj_quat_w)

    return torch.cat([rel_pos, rel_quat], dim=-1)


# -----------------------------------------------------------------------------
# Rewards
# -----------------------------------------------------------------------------

def finger_contact_reward(
    env: ManagerBasedRLEnv,
    sensor_name: str = "fingertip_contacts",
    threshold: float = 0.5,
) -> torch.Tensor:
    """Binary reward for any finger contact.

    Args:
        env: RL environment
        sensor_name: Name of contact sensor in scene
        threshold: Force threshold in Newtons

    Returns:
        Tensor of shape (num_envs,) with 1.0 for contact, 0.0 otherwise
    """
    if sensor_name not in env.scene.sensors:
        return torch.zeros(env.num_envs, device=env.device)

    sensor = env.scene[sensor_name]
    forces = sensor.data.net_forces_w  # (num_envs, num_bodies, 3)
    force_magnitudes = torch.norm(forces, dim=-1)  # (num_envs, num_bodies)
    any_contact = (force_magnitudes > threshold).any(dim=-1)
    return any_contact.float()


def multi_finger_contact(
    env: ManagerBasedRLEnv,
    sensor_name: str = "fingertip_contacts",
    min_fingers: int = 3,
    threshold: float = 0.5,
) -> torch.Tensor:
    """Reward for multiple fingers contacting object.

    Args:
        env: RL environment
        sensor_name: Name of contact sensor
        min_fingers: Minimum fingers required for reward
        threshold: Force threshold in Newtons

    Returns:
        Tensor of shape (num_envs,) with 1.0 if enough fingers contact
    """
    if sensor_name not in env.scene.sensors:
        return torch.zeros(env.num_envs, device=env.device)

    sensor = env.scene[sensor_name]
    forces = sensor.data.net_forces_w  # (num_envs, num_bodies, 3)
    force_magnitudes = torch.norm(forces, dim=-1)  # (num_envs, num_bodies)
    contact_count = (force_magnitudes > threshold).sum(dim=-1)
    return (contact_count >= min_fingers).float()


def object_lift_reward(
    env: ManagerBasedRLEnv,
    object_cfg: SceneEntityCfg = SceneEntityCfg("grasp_cylinder"),
    target_height: float = 0.05,
) -> torch.Tensor:
    """Continuous reward for lifting object.

    Args:
        env: RL environment
        object_cfg: Scene entity config for object
        target_height: Target lift height in meters

    Returns:
        Tensor of shape (num_envs,) with values in [0, 1]
    """
    obj: RigidObject = env.scene[object_cfg.name]

    # Track initial height
    buffer_name = "_obj_initial_z"
    if not hasattr(env, buffer_name):
        setattr(env, buffer_name, obj.data.root_pos_w[:, 2].clone())

    initial_z = getattr(env, buffer_name)
    current_z = obj.data.root_pos_w[:, 2]

    # Compute lift progress
    lift = torch.clamp((current_z - initial_z) / target_height, 0, 1)
    return lift


def grasp_hold_success(
    env: ManagerBasedRLEnv,
    object_cfg: SceneEntityCfg = SceneEntityCfg("grasp_cylinder"),
    lift_height: float = 0.05,
    hold_steps: int = 30,
) -> torch.Tensor:
    """Success reward when object is lifted and held stable.

    Args:
        env: RL environment
        object_cfg: Scene entity config for object
        lift_height: Required lift height
        hold_steps: Steps to hold for success

    Returns:
        Tensor of shape (num_envs,) with 1.0 for success
    """
    buffer_name = "_hold_count"
    if not hasattr(env, buffer_name):
        setattr(env, buffer_name, torch.zeros(env.num_envs, device=env.device, dtype=torch.int32))

    count = getattr(env, buffer_name)
    lift = object_lift_reward(env, object_cfg, lift_height)
    holding = lift > 0.9

    # Increment or reset counter
    count[:] = torch.where(holding, count + 1, torch.zeros_like(count))

    return (count >= hold_steps).float()


def object_dropped(
    env: ManagerBasedRLEnv,
    object_cfg: SceneEntityCfg = SceneEntityCfg("grasp_cylinder"),
    min_height: float = 0.7,
) -> torch.Tensor:
    """Penalty if object falls below threshold.

    Args:
        env: RL environment
        object_cfg: Scene entity config
        min_height: Height below which is considered dropped

    Returns:
        Tensor of shape (num_envs,) with 1.0 for dropped
    """
    obj: RigidObject = env.scene[object_cfg.name]
    dropped = obj.data.root_pos_w[:, 2] < min_height
    return dropped.float()


def grasp_force_reward(
    env: ManagerBasedRLEnv,
    sensor_name: str = "fingertip_contacts",
    target_force: float = 2.0,
    force_std: float = 1.0,
) -> torch.Tensor:
    """Reward for appropriate grasp force (not too weak, not too strong).

    Args:
        env: RL environment
        sensor_name: Contact sensor name
        target_force: Target total force
        force_std: Standard deviation for Gaussian reward

    Returns:
        Tensor of shape (num_envs,) with reward values
    """
    if sensor_name not in env.scene.sensors:
        return torch.zeros(env.num_envs, device=env.device)

    sensor = env.scene[sensor_name]
    forces = sensor.data.net_forces_w
    total_force = torch.norm(forces, dim=-1).sum(dim=-1)

    # Gaussian reward centered at target
    diff = (total_force - target_force) / force_std
    reward = torch.exp(-0.5 * diff * diff)
    return reward


# -----------------------------------------------------------------------------
# Terminations
# -----------------------------------------------------------------------------

def object_below_threshold(
    env: ManagerBasedRLEnv,
    object_cfg: SceneEntityCfg = SceneEntityCfg("grasp_cylinder"),
    min_height: float = 0.6,
) -> torch.Tensor:
    """Terminate if object falls below minimum height.

    Args:
        env: RL environment
        object_cfg: Scene entity config
        min_height: Termination height

    Returns:
        Boolean tensor of shape (num_envs,)
    """
    obj: RigidObject = env.scene[object_cfg.name]
    return obj.data.root_pos_w[:, 2] < min_height


def grasp_success_termination(
    env: ManagerBasedRLEnv,
    object_cfg: SceneEntityCfg = SceneEntityCfg("grasp_cylinder"),
    lift_height: float = 0.1,
    hold_steps: int = 50,
) -> torch.Tensor:
    """Terminate on successful grasp (lifted and held).

    Args:
        env: RL environment
        object_cfg: Scene entity config
        lift_height: Required lift height
        hold_steps: Steps to hold

    Returns:
        Boolean tensor of shape (num_envs,)
    """
    success = grasp_hold_success(env, object_cfg, lift_height, hold_steps)
    return success > 0.5


# -----------------------------------------------------------------------------
# Events
# -----------------------------------------------------------------------------

def reset_hold_counter(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
) -> None:
    """Reset the hold counter for specified environments."""
    buffer_name = "_hold_count"
    if not hasattr(env, buffer_name):
        setattr(env, buffer_name, torch.zeros(env.num_envs, device=env.device, dtype=torch.int32))
    count = getattr(env, buffer_name)
    count[env_ids] = 0


def reset_initial_object_height(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    object_cfg: SceneEntityCfg = SceneEntityCfg("grasp_cylinder"),
) -> None:
    """Reset tracked initial object height for specified environments."""
    buffer_name = "_obj_initial_z"
    obj: RigidObject = env.scene[object_cfg.name]
    current_z = obj.data.root_pos_w[:, 2]

    if not hasattr(env, buffer_name):
        setattr(env, buffer_name, current_z.clone())

    buf = getattr(env, buffer_name)
    buf[env_ids] = current_z[env_ids]


def hold_arm_at_pregrasp(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> None:
    """Hold arm joints at pre-grasp position (default targets).

    Used to lock the arm while training hand policy.
    """
    robot: Articulation = env.scene[asset_cfg.name]

    # Arm joint names to lock
    arm_joint_names = [
        "right_shoulder_pitch_joint",
        "right_shoulder_roll_joint",
        "right_shoulder_yaw_joint",
        "right_elbow_joint",
    ]

    # Find joint IDs
    joint_names = list(robot.data.joint_names)
    joint_ids = []
    for name in arm_joint_names:
        try:
            joint_ids.append(joint_names.index(name))
        except ValueError:
            continue

    if not joint_ids:
        return

    joint_ids_t = torch.tensor(joint_ids, device=env.device, dtype=torch.long)

    # Set arm joints to default positions
    default_pos = robot.data.default_joint_pos[env_ids][:, joint_ids_t]
    default_vel = robot.data.default_joint_vel[env_ids][:, joint_ids_t]

    robot.set_joint_position_target(default_pos, joint_ids=joint_ids_t, env_ids=env_ids)
    robot.set_joint_velocity_target(default_vel, joint_ids=joint_ids_t, env_ids=env_ids)


def place_object_near_hand(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    object_cfg: SceneEntityCfg = SceneEntityCfg("grasp_cylinder"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ee_body_name: str = "R_thumb_proximal",
    offset_local: tuple = (0.08, 0.0, -0.02),
) -> None:
    """Place object near the hand/end-effector for grasping.

    Args:
        env: RL environment
        env_ids: Environments to reset
        object_cfg: Object scene entity config
        robot_cfg: Robot scene entity config
        ee_body_name: End-effector body name
        offset_local: Offset from EE in local frame
    """
    from isaaclab.utils.math import quat_apply

    if env_ids.numel() == 0:
        return

    obj: RigidObject = env.scene[object_cfg.name]
    robot: Articulation = env.scene[robot_cfg.name]

    # Find EE body ID
    body_names = list(robot.data.body_names)
    try:
        ee_body_id = body_names.index(ee_body_name)
    except ValueError:
        print(f"[grasp_mdp] Warning: Body '{ee_body_name}' not found")
        return

    # Get EE world pose
    ee_pos_w = robot.data.body_pos_w[env_ids, ee_body_id]
    ee_quat_w = robot.data.body_quat_w[env_ids, ee_body_id]

    # Compute object position
    offset = torch.tensor(offset_local, device=env.device, dtype=ee_pos_w.dtype)
    offset = offset.unsqueeze(0).expand(env_ids.numel(), -1)
    offset_w = quat_apply(ee_quat_w, offset)
    obj_pos_w = ee_pos_w + offset_w

    # Set object pose (upright orientation)
    obj_quat_w = torch.tensor([1.0, 0.0, 0.0, 0.0], device=env.device, dtype=ee_pos_w.dtype)
    obj_quat_w = obj_quat_w.unsqueeze(0).expand(env_ids.numel(), -1)

    obj_pose = torch.cat([obj_pos_w, obj_quat_w], dim=-1)
    obj_vel = torch.zeros((env_ids.numel(), 6), device=env.device, dtype=ee_pos_w.dtype)

    obj.write_root_pose_to_sim(obj_pose, env_ids=env_ids)
    obj.write_root_velocity_to_sim(obj_vel, env_ids=env_ids)


__all__ = [
    # Observations
    "hand_joint_pos",
    "hand_joint_vel",
    "contact_forces",
    "object_pose_relative_to_hand",
    # Rewards
    "finger_contact_reward",
    "multi_finger_contact",
    "object_lift_reward",
    "grasp_hold_success",
    "object_dropped",
    "grasp_force_reward",
    # Terminations
    "object_below_threshold",
    "grasp_success_termination",
    # Events
    "reset_hold_counter",
    "reset_initial_object_height",
    "hold_arm_at_pregrasp",
    "place_object_near_hand",
]
