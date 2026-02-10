# Copyright (c) 2025, Unitree Robotics Co., Ltd. All Rights Reserved.
# License: Apache License, Version 2.0

"""
Termination conditions for Learn-to-Grasp task.
Handles success (cylinder in bin) and failure (cylinder dropped) conditions.
"""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

from isaaclab.managers import SceneEntityCfg

# Import phase constants - support both package and direct import modes
try:
    from .events import PHASE_RELEASE
except ImportError:
    import sys
    _events = sys.modules.get("learn_grasp_mdp_events")
    if _events:
        PHASE_RELEASE = _events.PHASE_RELEASE
    else:
        PHASE_RELEASE = 4  # Fallback constant


def _get_observations_module():
    """Get observations module, supporting both package and direct import."""
    try:
        from . import observations
        return observations
    except ImportError:
        import sys
        return sys.modules.get("learn_grasp_mdp_observations")


def cylinder_in_bin(
    env: ManagerBasedRLEnv,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    bin_center_x: float = 0.3,
    bin_center_y: float = -0.2,
    bin_height: float = 0.85,
    radius: float = 0.15,
) -> torch.Tensor:
    """Success termination: cylinder is placed in target bin.

    Args:
        env: Environment instance
        object_cfg: Scene entity config for cylinder
        bin_center_x: X coordinate of bin center
        bin_center_y: Y coordinate of bin center
        bin_height: Expected Z height in bin
        radius: Radius for "in bin" detection

    Returns:
        torch.Tensor: [num_envs] boolean - True if cylinder in bin
    """
    obj = env.scene[object_cfg.name]
    pos = obj.data.root_pos_w  # [num_envs, 3]
    vel = obj.data.root_lin_vel_w  # [num_envs, 3]

    # Distance from bin center (XY plane)
    dist_xy = torch.sqrt(
        (pos[:, 0] - bin_center_x) ** 2 + (pos[:, 1] - bin_center_y) ** 2
    )

    # Check position conditions
    in_bin_xy = dist_xy < radius
    in_bin_z = (pos[:, 2] > bin_height - 0.1) & (pos[:, 2] < bin_height + 0.2)

    # Check velocity (cylinder should be stationary)
    vel_magnitude = torch.norm(vel, dim=-1)
    is_stationary = vel_magnitude < 0.1

    # Must be in release phase
    if hasattr(env, "_task_phase"):
        in_release = env._task_phase == PHASE_RELEASE
    else:
        in_release = torch.ones(env.num_envs, dtype=torch.bool, device=env.device)

    success = in_bin_xy & in_bin_z & is_stationary & in_release

    return success


def cylinder_dropped(
    env: ManagerBasedRLEnv,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    min_height: float = 0.5,
) -> torch.Tensor:
    """Failure termination: cylinder fell below minimum height.

    Args:
        env: Environment instance
        object_cfg: Scene entity config for cylinder
        min_height: Minimum Z height before considered dropped

    Returns:
        torch.Tensor: [num_envs] boolean - True if cylinder dropped
    """
    obj = env.scene[object_cfg.name]
    height = obj.data.root_pos_w[:, 2]

    return height < min_height


def cylinder_out_of_bounds(
    env: ManagerBasedRLEnv,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    x_range: tuple = (-1.0, 1.0),
    y_range: tuple = (-1.0, 1.5),
) -> torch.Tensor:
    """Failure termination: cylinder moved too far from workspace.

    Args:
        env: Environment instance
        object_cfg: Scene entity config for cylinder
        x_range: Valid X coordinate range
        y_range: Valid Y coordinate range

    Returns:
        torch.Tensor: [num_envs] boolean - True if out of bounds
    """
    obj = env.scene[object_cfg.name]
    pos = obj.data.root_pos_w  # [num_envs, 3]

    out_x = (pos[:, 0] < x_range[0]) | (pos[:, 0] > x_range[1])
    out_y = (pos[:, 1] < y_range[0]) | (pos[:, 1] > y_range[1])

    return out_x | out_y


def cylinder_tipped_over(
    env: ManagerBasedRLEnv,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    max_tilt_cos: float = 0.7,
) -> torch.Tensor:
    """Failure termination: cylinder tipped over significantly.

    Args:
        env: Environment instance
        object_cfg: Scene entity config for cylinder
        max_tilt_cos: Minimum cosine of tilt angle (0.7 ~ 45 degrees)

    Returns:
        torch.Tensor: [num_envs] boolean - True if tipped over
    """
    obj = env.scene[object_cfg.name]
    quat = obj.data.root_quat_w  # [num_envs, 4] (w, x, y, z)

    # For a cylinder standing upright, local Z should align with world Z
    # Extract the Z component of the local up vector after rotation
    # Simplified: check w component of quaternion (for near-identity rotation)
    # More accurate: compute rotation matrix and check Z column

    # Using quaternion: local_z = R @ [0,0,1]
    # For quaternion (w, x, y, z):
    # R[2,2] = 1 - 2(x^2 + y^2) = cos(tilt from vertical)

    w, x, y, z = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]
    cos_tilt = 1.0 - 2.0 * (x * x + y * y)

    return cos_tilt < max_tilt_cos


def excessive_contact_force(
    env: ManagerBasedRLEnv,
    force_threshold: float = 50.0,
) -> torch.Tensor:
    """Failure termination: excessive contact force detected.

    Prevents policy from crushing the object.

    Args:
        env: Environment instance
        force_threshold: Maximum allowed contact force

    Returns:
        torch.Tensor: [num_envs] boolean - True if force exceeded
    """
    obs_mod = _get_observations_module()
    get_right_hand_contact_forces = obs_mod.get_right_hand_contact_forces

    forces = get_right_hand_contact_forces(env)  # [num_envs, 6]
    max_force = forces.max(dim=1).values

    return max_force > force_threshold


__all__ = [
    "cylinder_in_bin",
    "cylinder_dropped",
    "cylinder_out_of_bounds",
    "cylinder_tipped_over",
    "excessive_contact_force",
]
