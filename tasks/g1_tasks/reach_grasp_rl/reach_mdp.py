# Copyright (c) 2025, Unitree Robotics Co., Ltd. All Rights Reserved.
# License: Apache License, Version 2.0

"""MDP helper functions for G1 right arm reach and grasp task.

Adapted from examples-issac-lab/reach_mdp.py for right arm.
"""

from __future__ import annotations
from typing import Tuple
import torch

from isaaclab.assets import Articulation
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import subtract_frame_transforms


# --------------------------------------------------------------------------------------
# Utilities
# --------------------------------------------------------------------------------------

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


# --------------------------------------------------------------------------------------
# Observations
# --------------------------------------------------------------------------------------

def body_pos_in_root_frame(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Body position in robot root frame. Returns (num_envs, 3)."""
    asset: Articulation = env.scene[asset_cfg.name]
    body_id = _resolve_single_body_id(asset, asset_cfg)

    root_pos_w = asset.data.root_pos_w
    root_quat_w = asset.data.root_quat_w
    body_pos_w = asset.data.body_pos_w[:, body_id]
    body_quat_w = asset.data.body_quat_w[:, body_id]

    pos_b, _ = subtract_frame_transforms(root_pos_w, root_quat_w, body_pos_w, body_quat_w)
    return pos_b


# --------------------------------------------------------------------------------------
# Rewards
# --------------------------------------------------------------------------------------

def position_command_error(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, command_name: str) -> torch.Tensor:
    """L2 position error between EE and goal."""
    goal_b = env.command_manager.get_command(command_name)[:, 0:3]
    ee_pos_b = body_pos_in_root_frame(env, asset_cfg)
    return torch.linalg.norm(ee_pos_b - goal_b, dim=-1)


def position_command_error_tanh(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    command_name: str,
    std: float = 0.15,
    eps: float = 1e-6,
    squared: bool = False,
) -> torch.Tensor:
    """Bounded tracking reward in [0, 1]."""
    dist = position_command_error(env, asset_cfg=asset_cfg, command_name=command_name)
    x = dist / max(float(std), eps)
    if squared:
        x = x * x
    return 1.0 - torch.tanh(x)


def success_bonus(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    command_name: str,
    threshold: float = 0.05,
) -> torch.Tensor:
    """Returns 1 when within threshold of goal."""
    dist = position_command_error(env, asset_cfg=asset_cfg, command_name=command_name)
    done = dist < float(threshold)
    return done.to(dtype=env.command_manager.get_command(command_name).dtype)


# --------------------------------------------------------------------------------------
# Terminations
# --------------------------------------------------------------------------------------

def ee_lin_speed_w(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """End-effector linear speed in world frame."""
    asset: Articulation = env.scene[asset_cfg.name]
    body_id = _resolve_single_body_id(asset, asset_cfg)
    v = asset.data.body_lin_vel_w[:, body_id]
    return torch.linalg.norm(v, dim=-1)


def reached_and_still_dwell(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    command_name: str,
    pos_threshold: float = 0.05,
    speed_threshold: float = 0.03,
    dwell_steps: int = 10,
    buffer_name: str = "_ee_dwell_steps",
) -> torch.Tensor:
    """True when EE is within threshold and still for dwell_steps."""
    goal_pos = env.command_manager.get_command(command_name)[:, 0:3]
    ee_pos = body_pos_in_root_frame(env, asset_cfg)
    dist = torch.linalg.norm(ee_pos - goal_pos, dim=-1)
    speed = ee_lin_speed_w(env, asset_cfg)

    ok = (dist < float(pos_threshold)) & (speed < float(speed_threshold))

    if not hasattr(env, buffer_name):
        setattr(env, buffer_name, torch.zeros((env.num_envs,), device=env.device, dtype=torch.int32))
    ctr: torch.Tensor = getattr(env, buffer_name)

    ctr[:] = torch.where(ok, torch.clamp(ctr + 1, max=int(dwell_steps)), torch.zeros_like(ctr))
    return ctr >= int(dwell_steps)


# --------------------------------------------------------------------------------------
# Events / Resets
# --------------------------------------------------------------------------------------

def reset_named_buffer(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    buffer_name: str,
    dtype: str = "bool",
) -> None:
    """Create/reset a per-env buffer."""
    if not hasattr(env, buffer_name):
        if dtype == "bool":
            buf = torch.zeros((env.num_envs,), device=env.device, dtype=torch.bool)
        elif dtype == "int":
            buf = torch.zeros((env.num_envs,), device=env.device, dtype=torch.int32)
        elif dtype == "float":
            buf = torch.zeros((env.num_envs,), device=env.device, dtype=torch.float32)
        else:
            raise ValueError(f"Unsupported dtype: {dtype}")
        setattr(env, buffer_name, buf)

    buf: torch.Tensor = getattr(env, buffer_name)
    if buf.dtype == torch.bool:
        buf[env_ids] = False
    else:
        buf[env_ids] = 0


def hold_joints_at_default_targets(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> None:
    """Hold selected joints at default targets."""
    asset: Articulation = env.scene[asset_cfg.name]
    default_pos = asset.data.default_joint_pos[env_ids][:, asset_cfg.joint_ids]
    default_vel = asset.data.default_joint_vel[env_ids][:, asset_cfg.joint_ids]
    asset.set_joint_position_target(default_pos, joint_ids=asset_cfg.joint_ids, env_ids=env_ids)
    asset.set_joint_velocity_target(default_vel, joint_ids=asset_cfg.joint_ids, env_ids=env_ids)


def place_object_near_ee_goal(
    env,
    env_ids: torch.Tensor,
    object_cfg: SceneEntityCfg,
    robot_cfg: SceneEntityCfg,
    command_name: str,
    offset_root=(0.0, -0.02, -0.04),
):
    """Place a rigid object near the current EE goal (goal is expressed in robot root frame).

    Matches example: places cylinder at goal position with offset.
    """
    from isaaclab.utils.math import quat_apply

    obj = env.scene[object_cfg.name]
    robot = env.scene[robot_cfg.name]

    # Command is in robot root frame: [x, y, z, qw, qx, qy, qz]
    cmd = env.command_manager.get_command(command_name)[env_ids]
    goal_pos_root = cmd[:, 0:3]

    root_pos_w = robot.data.root_pos_w[env_ids]
    root_quat_w = robot.data.root_quat_w[env_ids]

    # root -> world
    goal_pos_w = root_pos_w + quat_apply(root_quat_w, goal_pos_root)

    # offset is also specified in robot root frame
    off = torch.tensor(offset_root, device=env.device, dtype=goal_pos_root.dtype).view(1, 3).repeat(env_ids.numel(), 1)
    off_w = quat_apply(root_quat_w, off)

    obj_pos_w = goal_pos_w + off_w
    obj_quat_w = torch.tensor([1.0, 0.0, 0.0, 0.0], device=env.device, dtype=goal_pos_root.dtype).view(1, 4).repeat(env_ids.numel(), 1)

    obj.write_root_pose_to_sim(torch.cat([obj_pos_w, obj_quat_w], dim=-1), env_ids=env_ids)

    # Record spawn position for later penalties/terminations
    if not hasattr(env, "_cyl_spawn_pos_w"):
        env._cyl_spawn_pos_w = torch.zeros((env.num_envs, 3), device=env.device, dtype=obj_pos_w.dtype)
    env._cyl_spawn_pos_w[env_ids] = obj_pos_w

    # Zero velocity at spawn
    obj.write_root_velocity_to_sim(torch.zeros((env_ids.numel(), 6), device=env.device, dtype=goal_pos_root.dtype), env_ids=env_ids)


def place_object_near_ee_goal_once(
    env,
    env_ids: torch.Tensor,
    object_cfg: SceneEntityCfg,
    robot_cfg: SceneEntityCfg,
    command_name: str,
    offset_root=(0.0, -0.02, -0.04),
    placed_buffer_name: str = "_cyl_placed",
):
    """Place object near the current EE goal exactly once per episode."""
    if not hasattr(env, placed_buffer_name):
        setattr(env, placed_buffer_name, torch.zeros((env.num_envs,), device=env.device, dtype=torch.bool))
    placed = getattr(env, placed_buffer_name)

    todo = env_ids[~placed[env_ids]]
    if todo.numel() == 0:
        return

    place_object_near_ee_goal(
        env, todo,
        object_cfg=object_cfg,
        robot_cfg=robot_cfg,
        command_name=command_name,
        offset_root=offset_root,
    )
    placed[todo] = True


def hold_hand_open_then_close_on_reach(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    ee_cfg: SceneEntityCfg,
    hand_joint_cfg: SceneEntityCfg,
    command_name: str,
    reach_threshold: float = 0.05,
    stop_only: bool = False,
    min_steps_before_stall: int = 25,
    stall_vel_threshold: float = 0.03,
    stall_pos_threshold: float = 5e-4,
    stall_near_threshold: float = 0.08,
    stall_steps: int = 10,
    stall_buffer_name: str = "_ee_still_steps",
    step_buffer_name: str = "_ee_total_steps",
    prev_pos_buffer_name: str = "_ee_prev_pos_b",
    open_pos=None,
    close_pos=None,
    close_pos_is_degrees: bool = True,
    latch_buffer_name: str = "_hand_closed",
    close_ramp_steps: int = 20,
    ramp_buffer_name: str = "_hand_close_ramp",
) -> None:
    """Hold hand open until reach/stall, then close (latched)."""

    robot: Articulation = env.scene[hand_joint_cfg.name]
    ee_body_id = _resolve_single_body_id(robot, ee_cfg)

    # Buffers
    if not hasattr(env, latch_buffer_name):
        setattr(env, latch_buffer_name, torch.zeros(env.num_envs, device=env.device, dtype=torch.bool))
    hand_closed = getattr(env, latch_buffer_name)

    if not hasattr(env, ramp_buffer_name):
        setattr(env, ramp_buffer_name, torch.zeros(env.num_envs, device=env.device, dtype=torch.int32))
    close_ramp = getattr(env, ramp_buffer_name)

    if not hasattr(env, step_buffer_name):
        setattr(env, step_buffer_name, torch.zeros(env.num_envs, device=env.device, dtype=torch.int32))
    ee_steps = getattr(env, step_buffer_name)

    if not hasattr(env, stall_buffer_name):
        setattr(env, stall_buffer_name, torch.zeros(env.num_envs, device=env.device, dtype=torch.int32))
    still_steps = getattr(env, stall_buffer_name)

    ee_pos_all = body_pos_in_root_frame(env, ee_cfg)
    if not hasattr(env, prev_pos_buffer_name):
        setattr(env, prev_pos_buffer_name, ee_pos_all.clone())
    ee_prev = getattr(env, prev_pos_buffer_name)

    ee_steps[env_ids] += 1

    # Resolve joint ids
    joint_names = getattr(hand_joint_cfg, "joint_names", None)
    if not joint_names:
        raise ValueError("hand_joint_cfg.joint_names must be provided.")

    name_to_id = {n: i for i, n in enumerate(robot.data.joint_names)}
    joint_ids = torch.tensor([name_to_id[n] for n in joint_names], device=env.device, dtype=torch.long)
    J = len(joint_names)

    def _to_targets(x, *, degrees: bool):
        if x is None:
            return None
        if isinstance(x, dict):
            vals = [float(x[n]) for n in joint_names]
        elif isinstance(x, (float, int)):
            vals = [float(x)] * J
        else:
            vals = list(x)
        t = torch.tensor(vals, device=env.device, dtype=robot.data.default_joint_pos.dtype)
        if degrees:
            t = torch.deg2rad(t)
        return t

    open_t = _to_targets(open_pos, degrees=close_pos_is_degrees) if open_pos is not None else None
    close_t = _to_targets(close_pos, degrees=close_pos_is_degrees) if close_pos is not None else None
    if open_t is None:
        open_t = torch.zeros((J,), device=env.device, dtype=robot.data.default_joint_pos.dtype)
    if close_t is None:
        close_t = open_t.clone()

    # Distance to goal
    goal_cmd = env.command_manager.get_command(command_name)
    goal_pos = goal_cmd[:, 0:3]
    ee_pos = ee_pos_all
    dist = torch.norm(ee_pos - goal_pos, dim=-1)

    # Stillness
    ee_lin_vel = robot.data.body_lin_vel_w[:, ee_body_id, :]
    ee_lin_speed = torch.norm(ee_lin_vel, dim=-1)
    pos_delta = torch.norm(ee_pos - ee_prev, dim=-1)
    ee_prev[env_ids] = ee_pos[env_ids]

    is_still = (ee_lin_speed < stall_vel_threshold) | (pos_delta < stall_pos_threshold)
    still_steps[env_ids] = torch.where(is_still[env_ids], still_steps[env_ids] + 1, torch.zeros_like(still_steps[env_ids]))

    enough_steps = ee_steps >= min_steps_before_stall
    stalled = enough_steps & (still_steps >= stall_steps)

    reached = dist <= reach_threshold
    if stop_only:
        trigger = stalled
    else:
        trigger = reached | (stalled & (dist <= stall_near_threshold))

    # Close hand
    to_close = trigger & (~hand_closed)
    hand_closed[to_close] = True

    closing = hand_closed.clone()
    close_ramp[closing] = torch.clamp(close_ramp[closing] + 1, max=close_ramp_steps)

    alpha = (close_ramp.float() / float(max(close_ramp_steps, 1))).unsqueeze(-1)
    open_targets = open_t.unsqueeze(0).repeat(env.num_envs, 1)
    close_targets = close_t.unsqueeze(0).repeat(env.num_envs, 1)
    targets = torch.where(closing.unsqueeze(-1), open_targets + alpha * (close_targets - open_targets), open_targets)

    targets_ids = targets[env_ids]
    robot.set_joint_position_target(targets_ids, joint_ids=joint_ids, env_ids=env_ids)


def reset_ee_prev_pos_buffer(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    ee_cfg: SceneEntityCfg,
    buffer_name: str = "_ee_prev_pos_b",
) -> None:
    """Reset prev EE pos buffer."""
    ee_pos_all = body_pos_in_root_frame(env, ee_cfg)
    if not hasattr(env, buffer_name):
        buf = torch.zeros((env.num_envs, 3), device=env.device, dtype=ee_pos_all.dtype)
        setattr(env, buffer_name, buf)
    buf = getattr(env, buffer_name)
    buf[env_ids] = ee_pos_all[env_ids]


def ensure_goal_reachable_box_sphere(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    command_name: str,
    shoulder_cfg: SceneEntityCfg,
    x_range: Tuple[float, float] = (0.3, 0.4),
    y_range: Tuple[float, float] = (-0.05, 0.3),
    z_range: Tuple[float, float] = (0.05, 0.3),
    r_min: float = 0.12,
    r_max: float = 0.48,
    max_tries: int = 80,
) -> None:
    """Ensure commanded goal is inside AABB and spherical shell around shoulder.

    Validates that goals are kinematically reachable by the arm.
    If invalid, resamples within the box until valid.
    """
    if env_ids.numel() == 0:
        return

    cmd = env.command_manager.get_command(command_name)
    if cmd.shape[-1] < 3:
        raise ValueError(f"Command '{command_name}' must have at least xyz.")

    robot: Articulation = env.scene[shoulder_cfg.name]
    shoulder_id = _resolve_single_body_id(robot, shoulder_cfg)

    # Shoulder position in root frame
    root_pos_w = robot.data.root_pos_w[env_ids]
    root_quat_w = robot.data.root_quat_w[env_ids]
    shoulder_pos_w = robot.data.body_pos_w[env_ids, shoulder_id]
    shoulder_quat_w = robot.data.body_quat_w[env_ids, shoulder_id]
    shoulder_pos_b, _ = subtract_frame_transforms(root_pos_w, root_quat_w, shoulder_pos_w, shoulder_quat_w)

    goal_pos = cmd[env_ids, 0:3]

    x0, x1 = map(float, x_range)
    y0, y1 = map(float, y_range)
    z0, z1 = map(float, z_range)

    in_box = (
        (goal_pos[:, 0] >= x0) & (goal_pos[:, 0] <= x1) &
        (goal_pos[:, 1] >= y0) & (goal_pos[:, 1] <= y1) &
        (goal_pos[:, 2] >= z0) & (goal_pos[:, 2] <= z1)
    )
    d = torch.linalg.norm(goal_pos - shoulder_pos_b, dim=-1)
    in_shell = (d >= float(r_min)) & (d <= float(r_max))

    ok = in_box & in_shell
    if ok.all():
        return

    bad_local = (~ok).nonzero(as_tuple=False).squeeze(-1)
    bad_env_ids = env_ids[bad_local]
    B = int(bad_env_ids.numel())

    shoulder_bad = shoulder_pos_b[bad_local]
    device, dtype = cmd.device, cmd.dtype
    new_pos = torch.empty((B, 3), device=device, dtype=dtype)
    accepted = torch.zeros((B,), device=device, dtype=torch.bool)

    # Rejection sampling
    for _ in range(int(max_tries)):
        m = (~accepted).nonzero(as_tuple=False).squeeze(-1)
        if m.numel() == 0:
            break

        n = int(m.numel())
        samp = torch.empty((n, 3), device=device, dtype=dtype)
        samp[:, 0].uniform_(x0, x1)
        samp[:, 1].uniform_(y0, y1)
        samp[:, 2].uniform_(z0, z1)

        dd = torch.linalg.norm(samp - shoulder_bad[m], dim=-1)
        good = (dd >= float(r_min)) & (dd <= float(r_max))
        if good.any():
            good_idx = m[good]
            new_pos[good_idx] = samp[good]
            accepted[good_idx] = True

    # Fallback: random direction at mid radius + clamp into box
    if not accepted.all():
        m = (~accepted).nonzero(as_tuple=False).squeeze(-1)
        dirs = torch.randn((int(m.numel()), 3), device=device, dtype=dtype)
        dirs = dirs / (torch.linalg.norm(dirs, dim=-1, keepdim=True) + 1e-8)
        r_mid = 0.5 * (float(r_min) + float(r_max))
        p = shoulder_bad[m] + r_mid * dirs
        p[:, 0] = torch.clamp(p[:, 0], x0, x1)
        p[:, 1] = torch.clamp(p[:, 1], y0, y1)
        p[:, 2] = torch.clamp(p[:, 2], z0, z1)
        new_pos[m] = p

    # Write back to command buffer
    if hasattr(env.command_manager, "set_command"):
        cmd_new = cmd.clone()
        cmd_new[bad_env_ids, 0:3] = new_pos
        env.command_manager.set_command(command_name, cmd_new)
    else:
        cmd[bad_env_ids, 0:3] = new_pos


def switch_goal_position_after_hand_close(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    command_name: str,
    target_pos=(0.3, 0.05, 0.25),
    latch_buffer_name: str = "_hand_closed",
    wait_for_close_ramp: bool = True,
    close_ramp_buffer_name: str = "_hand_close_ramp",
    close_ramp_steps: int = 30,
    once: bool = True,
    switched_buffer_name: str = "_goal_switched",
) -> None:
    """After hand closes, switch goal to a new position (for lifting)."""
    if not hasattr(env, latch_buffer_name):
        return

    hand_closed = getattr(env, latch_buffer_name)
    mask = hand_closed[env_ids].clone()

    if wait_for_close_ramp:
        if not hasattr(env, close_ramp_buffer_name):
            return
        ramp = getattr(env, close_ramp_buffer_name)
        mask &= (ramp[env_ids] >= int(close_ramp_steps))

    if not mask.any():
        return

    if once:
        if not hasattr(env, switched_buffer_name):
            setattr(env, switched_buffer_name, torch.zeros((env.num_envs,), device=env.device, dtype=torch.bool))
        switched = getattr(env, switched_buffer_name)
        mask &= ~switched[env_ids]
        if not mask.any():
            return

    ids = env_ids[mask.nonzero(as_tuple=False).squeeze(-1)]

    cmd = env.command_manager.get_command(command_name)
    tgt = torch.tensor(target_pos, device=cmd.device, dtype=cmd.dtype).view(1, 3).repeat(ids.numel(), 1)

    if hasattr(env.command_manager, "set_command"):
        cmd_new = cmd.clone()
        cmd_new[ids, 0:3] = tgt
        env.command_manager.set_command(command_name, cmd_new)
    else:
        cmd[ids, 0:3] = tgt

    if once:
        switched[ids] = True


def open_hand_once_when_at_current_goal(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    ee_cfg: SceneEntityCfg,
    hand_joint_cfg: SceneEntityCfg,
    command_name: str,
    goal_switched_buffer_name: str = "_goal_switched",
    reach_threshold: float = 0.05,
    opened_buffer_name: str = "_hand_opened_drop",
    open_ramp_steps: int = 20,
    ramp_buffer_name: str = "_hand_open_ramp",
    start_buffer_name: str = "_hand_open_start",
    open_pos=None,
    open_pos_is_degrees: bool = True,
) -> None:
    """After the goal switches (goal #2 active), open the hand once when EE reaches the current goal.

    This is used to release the object above a target (e.g., a bin).
    """

    if not hasattr(env, goal_switched_buffer_name):
        return
    goal_switched_all = getattr(env, goal_switched_buffer_name)

    switched_mask = goal_switched_all[env_ids].clone()
    if not switched_mask.any():
        return

    goal_pos_all = env.command_manager.get_command(command_name)[:, 0:3]
    ee_pos_all = body_pos_in_root_frame(env, ee_cfg)

    dist = torch.norm(ee_pos_all - goal_pos_all, dim=-1)
    reach_mask = dist[env_ids] <= reach_threshold
    mask = switched_mask & reach_mask
    if not mask.any():
        return

    if not hasattr(env, opened_buffer_name):
        setattr(env, opened_buffer_name, torch.zeros(env.num_envs, device=env.device, dtype=torch.bool))
    opened = getattr(env, opened_buffer_name)

    mask &= ~opened[env_ids]
    if not mask.any():
        return

    ids = env_ids[mask.nonzero(as_tuple=False).squeeze(-1)]

    robot: Articulation = env.scene[hand_joint_cfg.name]
    joint_names = getattr(hand_joint_cfg, "joint_names", None)
    if not joint_names:
        raise ValueError("hand_joint_cfg.joint_names must be provided.")

    name_to_id = {n: i for i, n in enumerate(robot.data.joint_names)}
    joint_ids = torch.tensor([name_to_id[n] for n in joint_names], device=env.device, dtype=torch.long)
    J = len(joint_names)

    def _to_targets(x, *, degrees: bool):
        if x is None:
            vals = [0.0] * J
        elif isinstance(x, dict):
            vals = [float(x[n]) for n in joint_names]
        elif isinstance(x, (float, int)):
            vals = [float(x)] * J
        else:
            vals = list(x)
            if len(vals) != J:
                raise ValueError(f"Target length mismatch: expected {J}, got {len(vals)}")
        t = torch.tensor(vals, device=env.device, dtype=robot.data.default_joint_pos.dtype)
        if degrees:
            t = torch.deg2rad(t)
        return t

    open_t = _to_targets(open_pos, degrees=open_pos_is_degrees)

    # buffers for ramp + per-env start pos
    if not hasattr(env, ramp_buffer_name):
        setattr(env, ramp_buffer_name, torch.zeros(env.num_envs, device=env.device, dtype=torch.int32))
    ramp = getattr(env, ramp_buffer_name)

    # start pos buffer must be (N,J)
    if not hasattr(env, start_buffer_name):
        setattr(env, start_buffer_name, torch.zeros((env.num_envs, J), device=env.device, dtype=robot.data.default_joint_pos.dtype))
    start_buf = getattr(env, start_buffer_name)

    # capture starting positions on first trigger
    cur_pos = robot.data.joint_pos[:, joint_ids]  # (N,J)
    start_buf[ids] = cur_pos[ids]
    ramp[ids] = 0

    # run ramp for all envs that are in "opening" state but not finished yet
    opening = (~opened) & goal_switched_all
    ramp[opening] = torch.clamp(ramp[opening] + 1, max=open_ramp_steps)

    alpha = (ramp.float() / float(max(open_ramp_steps, 1))).unsqueeze(-1)
    targets = start_buf + alpha * (open_t.unsqueeze(0) - start_buf)

    robot.set_joint_position_target(targets[env_ids], joint_ids=joint_ids, env_ids=env_ids)

    # mark opened when ramp finished for these ids
    done = ramp[ids] >= int(open_ramp_steps)
    opened[ids[done]] = True
