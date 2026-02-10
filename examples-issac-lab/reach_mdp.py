"""reach_mdp.py

Custom MDP helper functions for the G1 reach task.

This file is a cleaned, minimal subset based on your uploaded
`reach_mdp (3F).py`, keeping the same core ideas but removing duplicate/
conflicting definitions.

It provides exactly the functions referenced by your env cfg:
  - Events: ensure_goal_reachable_box_sphere, hold_joints_at_default_targets, reset_named_buffer
  - Observations: body_pos_in_root_frame
  - Rewards: position_command_error, position_command_error_tanh, success_bonus
  - Terminations: reached_and_still_dwell

If you later re-enable the more advanced grasp / contact logic from the 3F file,
you can paste those helpers below this minimal block.
"""

from __future__ import annotations

from typing import Tuple

import torch

from isaaclab.assets import Articulation
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import subtract_frame_transforms


# --------------------------------------------------------------------------------------
# Small utilities
# --------------------------------------------------------------------------------------

def _to_1d_list(x) -> list:
    """Convert SceneEntityCfg fields (str | list[str]) into a python list."""
    if x is None:
        return []
    if isinstance(x, (list, tuple)):
        return list(x)
    return [x]


def _resolve_single_body_id(asset: Articulation, asset_cfg: SceneEntityCfg) -> int:
    """Resolve a single body index from a SceneEntityCfg.

    In IsaacLab, SceneEntityCfg.body_ids may be a list[int] OR a slice(None) sentinel
    (meaning "all bodies") before manager initialization resolves names -> ids.
    We handle both cases by falling back to body_names when needed.
    """
    body_ids = getattr(asset_cfg, "body_ids", None)

    # Already resolved case: list/tuple/tensor
    if body_ids is not None and not isinstance(body_ids, slice):
        if isinstance(body_ids, torch.Tensor):
            return int(body_ids.flatten()[0].item())
        return int(list(body_ids)[0])

    # Fallback: resolve from body_names (must be exactly one)
    body_names = _to_1d_list(getattr(asset_cfg, "body_names", None))
    if len(body_names) != 1:
        raise ValueError(
            "Expected SceneEntityCfg with exactly one body (body_names=[...]) "
            f"or resolved body_ids=[...]. Got body_names={body_names}, body_ids={body_ids}."
        )

    name0 = body_names[0]

    # Preferred: Articulation.find_bodies (returns (ids, names) in most builds)
    if hasattr(asset, "find_bodies"):
        res = asset.find_bodies([name0])
        ids = res[0] if isinstance(res, tuple) else res
        return int(ids[0])

    # Fallback: look up in asset.data.body_names
    if hasattr(asset, "data") and hasattr(asset.data, "body_names"):
        return int(list(asset.data.body_names).index(name0))

    raise AttributeError("Cannot resolve body id: missing find_bodies and data.body_names.")


# --------------------------------------------------------------------------------------
# Observations
# --------------------------------------------------------------------------------------

def body_pos_in_root_frame(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Body position expressed in the asset root frame.

    UniformPoseCommand samples goals in the robot base/root frame, so this keeps frames consistent.
    Returns: (num_envs, 3)
    """
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
    """L2 position error between current body pos and commanded goal (robot root frame)."""
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
    """Stable bounded tracking reward in [0, 1].

    r = 1 - tanh(dist / std)
    Optionally: r = 1 - tanh((dist/std)^2) for sharper near-goal shaping.
    """
    dist = position_command_error(env, asset_cfg=asset_cfg, command_name=command_name)
    x = dist / max(float(std), eps)
    if squared:
        x = x * x
    return 1.0 - torch.tanh(x)


def _reached_ee_position(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    command_name: str,
    threshold: float,
) -> torch.Tensor:
    """Boolean mask per-env for reaching goal position."""
    return position_command_error(env, asset_cfg=asset_cfg, command_name=command_name) < float(threshold)


def success_bonus(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    command_name: str,
    threshold: float = 0.05,
) -> torch.Tensor:
    """Returns 1 when within threshold of goal, else 0 (as float tensor)."""
    done = _reached_ee_position(env, asset_cfg=asset_cfg, command_name=command_name, threshold=threshold)
    return done.to(dtype=env.command_manager.get_command(command_name).dtype)


# --------------------------------------------------------------------------------------
# Terminations
# --------------------------------------------------------------------------------------

def ee_lin_speed_w(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """End-effector linear speed in world frame (m/s)."""
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
    """True when EE is within pos_threshold and speed < speed_threshold for dwell_steps consecutive steps."""
    goal_pos = env.command_manager.get_command(command_name)[:, 0:3]
    ee_pos = body_pos_in_root_frame(env, asset_cfg)
    dist = torch.linalg.norm(ee_pos - goal_pos, dim=-1)
    speed = ee_lin_speed_w(env, asset_cfg)

    ok = (dist < float(pos_threshold)) & (speed < float(speed_threshold))

    if not hasattr(env, buffer_name):
        setattr(env, buffer_name, torch.zeros((env.num_envs,), device=env.device, dtype=torch.int32))
    ctr: torch.Tensor = getattr(env, buffer_name)

    # update for all envs (termination terms are typically evaluated for all envs)
    ctr[:] = torch.where(ok, torch.clamp(ctr + 1, max=int(dwell_steps)), torch.zeros_like(ctr))
    return ctr >= int(dwell_steps)


# --------------------------------------------------------------------------------------
# Events / resets
# --------------------------------------------------------------------------------------

def reset_named_buffer(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    buffer_name: str,
    dtype: str = "bool",
) -> None:
    """Create/reset a per-env buffer.

    dtype: "bool" | "int" | "float"
    """
    if not hasattr(env, buffer_name):
        if dtype == "bool":
            buf = torch.zeros((env.num_envs,), device=env.device, dtype=torch.bool)
        elif dtype == "int":
            buf = torch.zeros((env.num_envs,), device=env.device, dtype=torch.int32)
        elif dtype == "float":
            buf = torch.zeros((env.num_envs,), device=env.device, dtype=torch.float32)
        else:
            raise ValueError(f"Unsupported dtype for reset_named_buffer: {dtype}")
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
    """Hard-hold selected joints at their default targets each step (use as an interval event)."""
    asset: Articulation = env.scene[asset_cfg.name]

    default_pos = asset.data.default_joint_pos[env_ids][:, asset_cfg.joint_ids]
    default_vel = asset.data.default_joint_vel[env_ids][:, asset_cfg.joint_ids]

    asset.set_joint_position_target(default_pos, joint_ids=asset_cfg.joint_ids, env_ids=env_ids)
    asset.set_joint_velocity_target(default_vel, joint_ids=asset_cfg.joint_ids, env_ids=env_ids)


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
    """Ensure the commanded goal stays inside:
      1) an AABB defined by x/y/z ranges (in robot root frame)
      2) a spherical shell [r_min, r_max] centered at the shoulder (also in root frame)

    If the current command is invalid for some envs, resample within the box until valid.
    """
    if env_ids.numel() == 0:
        return

    cmd = env.command_manager.get_command(command_name)
    if cmd.shape[-1] < 3:
        raise ValueError(f"Command '{command_name}' must have at least xyz (got shape {tuple(cmd.shape)}).")

    robot: Articulation = env.scene[shoulder_cfg.name]
    shoulder_id = _resolve_single_body_id(robot, shoulder_cfg)

    # Shoulder position in root frame (per selected env)
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

    # rejection sampling
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

    # fallback: random direction at mid radius + clamp into box
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

    # write back to command buffer
    if hasattr(env.command_manager, "set_command"):
        cmd_new = cmd.clone()
        cmd_new[bad_env_ids, 0:3] = new_pos
        env.command_manager.set_command(command_name, cmd_new)
    else:
        cmd[bad_env_ids, 0:3] = new_pos



def place_object_near_ee_goal(
    env,
    env_ids: torch.Tensor,
    object_cfg: SceneEntityCfg,
    robot_cfg: SceneEntityCfg,
    command_name: str,
    offset_root=(0.0, -0.02, -0.04),
):
    """Place a rigid object near the current EE goal (goal is expressed in robot root frame)."""
    # local import so you don't have to modify your file's top-level imports
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
    off = torch.tensor(
        offset_root, device=env.device, dtype=goal_pos_root.dtype
    ).view(1, 3).repeat(env_ids.numel(), 1)
    off_w = quat_apply(root_quat_w, off)

    obj_pos_w = goal_pos_w + off_w
    obj_quat_w = torch.tensor(
        [1.0, 0.0, 0.0, 0.0], device=env.device, dtype=goal_pos_root.dtype
    ).view(1, 4).repeat(env_ids.numel(), 1)

    obj.write_root_pose_to_sim(torch.cat([obj_pos_w, obj_quat_w], dim=-1), env_ids=env_ids)

    # (optional but useful) record spawn position for later penalties/terminations
    if not hasattr(env, "_cyl_spawn_pos_w"):
        env._cyl_spawn_pos_w = torch.zeros((env.num_envs, 3), device=env.device, dtype=obj_pos_w.dtype)
    env._cyl_spawn_pos_w[env_ids] = obj_pos_w

    # zero velocity at spawn
    obj.write_root_velocity_to_sim(
        torch.zeros((env_ids.numel(), 6), device=env.device, dtype=goal_pos_root.dtype),
        env_ids=env_ids,
    )


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
        env,
        todo,
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
    # "stopped moving" logic (optional)
    stop_only: bool = False,
    min_steps_before_stall: int = 25,
    # stillness thresholds (either can trigger stillness)
    stall_vel_threshold: float = 0.03,
    stall_pos_threshold: float = 5e-4,
    # near-goal gating when stop_only=False
    stall_near_threshold: float = 0.08,
    stall_steps: int = 10,
    stall_buffer_name: str = "_ee_still_steps",
    step_buffer_name: str = "_ee_total_steps",
    prev_pos_buffer_name: str = "_ee_prev_pos_b",
    # joint targets
    open_pos=None,
    close_pos=None,
    close_pos_is_degrees: bool = True,
    # latch state
    latch_buffer_name: str = "_hand_closed",
    # close ramp (to avoid PhysX "teleport"/pop from instant penetration)
    close_ramp_steps: int = 20,
    ramp_buffer_name: str = "_hand_close_ramp",
) -> None:
    """Hold selected hand joints at `open_pos` until a trigger condition, then close (latched).

    Key fixes vs earlier versions:
      1) Joint ordering is guaranteed: joint_ids resolved in exact order of `hand_joint_cfg.joint_names`.
      2) Supports open/close targets as None | scalar | sequence | dict{name->val}.
      3) Close ramp to avoid physics popping.
    """

    robot: Articulation = env.scene[hand_joint_cfg.name]

    # Resolve EE body id (single) on the robot
    ee_body_id = _resolve_single_body_id(robot, ee_cfg)

    # --- buffers: latch, ramp, step counter, still counter, prev pos ---
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

    # prev pos buffer used for stillness
    ee_pos_all = body_pos_in_root_frame(env, ee_cfg)  # (N,3)
    if not hasattr(env, prev_pos_buffer_name):
        setattr(env, prev_pos_buffer_name, ee_pos_all.clone())
    ee_prev = getattr(env, prev_pos_buffer_name)

    # increment per-step counter for these envs
    ee_steps[env_ids] += 1

    # --- resolve joint ids in the exact order given ---
    joint_names = getattr(hand_joint_cfg, "joint_names", None)
    if not joint_names:
        raise ValueError("hand_joint_cfg.joint_names must be provided.")

    name_to_id = {n: i for i, n in enumerate(robot.data.joint_names)}
    try:
        joint_ids = torch.tensor([name_to_id[n] for n in joint_names], device=env.device, dtype=torch.long)
    except KeyError as e:
        raise KeyError(f"Joint name not found on robot: {e}. Check hand_joint_cfg.joint_names") from e

    J = len(joint_names)

    def _to_targets(x, *, degrees: bool):
        """Convert x to a (J,) torch tensor. x can be None|scalar|seq(J)|dict(name->val)."""
        if x is None:
            return None
        if isinstance(x, dict):
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

    open_t = _to_targets(open_pos, degrees=close_pos_is_degrees) if open_pos is not None else None
    close_t = _to_targets(close_pos, degrees=close_pos_is_degrees) if close_pos is not None else None
    if open_t is None:
        open_t = torch.zeros((J,), device=env.device, dtype=robot.data.default_joint_pos.dtype)
    if close_t is None:
        close_t = open_t.clone()

    # --- compute distance to goal (current command) ---
    goal_cmd = env.command_manager.get_command(command_name)  # (N, 7) [x,y,z,qw,qx,qy,qz] in root frame
    goal_pos = goal_cmd[:, 0:3]
    ee_pos = ee_pos_all
    dist = torch.norm(ee_pos - goal_pos, dim=-1)  # (N,)

    # --- stillness / stop logic ---
    ee_lin_vel = robot.data.body_lin_vel_w[:, ee_body_id, :]  # (N,3) world
    ee_lin_speed = torch.norm(ee_lin_vel, dim=-1)

    pos_delta = torch.norm(ee_pos - ee_prev, dim=-1)  # (N,)
    ee_prev[env_ids] = ee_pos[env_ids]

    is_still = (ee_lin_speed < stall_vel_threshold) | (pos_delta < stall_pos_threshold)

    # count consecutive still steps
    still_steps[env_ids] = torch.where(
        is_still[env_ids],
        still_steps[env_ids] + 1,
        torch.zeros_like(still_steps[env_ids]),
    )

    # gate: require some steps before declaring stall
    enough_steps = ee_steps >= min_steps_before_stall
    stalled = enough_steps & (still_steps >= stall_steps)

    # trigger conditions
    reached = dist <= reach_threshold
    if stop_only:
        trigger = stalled
    else:
        trigger = reached | (stalled & (dist <= stall_near_threshold))

    # --- set joint targets ---
    # open for not-yet-triggered, close (latched) for triggered
    to_close = trigger & (~hand_closed)
    hand_closed[to_close] = True

    # ramp progresses only for closed envs
    closing = hand_closed.clone()
    close_ramp[closing] = torch.clamp(close_ramp[closing] + 1, max=close_ramp_steps)

    # compute per-env interpolated target
    alpha = (close_ramp.float() / float(max(close_ramp_steps, 1))).unsqueeze(-1)  # (N,1)
    open_targets = open_t.unsqueeze(0).repeat(env.num_envs, 1)  # (N,J)
    close_targets = close_t.unsqueeze(0).repeat(env.num_envs, 1)  # (N,J)
    targets = torch.where(
        closing.unsqueeze(-1),
        open_targets + alpha * (close_targets - open_targets),
        open_targets,
    )

    # apply only to env_ids
    targets_ids = targets[env_ids]  # (len(env_ids),J)
    robot.set_joint_position_target(targets_ids, joint_ids=joint_ids, env_ids=env_ids)


def reset_ee_prev_pos_buffer(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    ee_cfg: SceneEntityCfg,
    buffer_name: str = "_ee_prev_pos_b",
) -> None:
    """Reset prev EE pos buffer used by hold_hand_open_then_close_on_reach."""
    ee_pos_all = body_pos_in_root_frame(env, ee_cfg)  # (N,3)
    if not hasattr(env, buffer_name):
        buf = torch.zeros((env.num_envs, 3), device=env.device, dtype=ee_pos_all.dtype)
        setattr(env, buffer_name, buf)
    buf = getattr(env, buffer_name)
    buf[env_ids] = ee_pos_all[env_ids]


def switch_goal_position_after_hand_close(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    command_name: str,
    target_pos=(0.4, 0.1, 0.4),
    latch_buffer_name: str = "_hand_closed",
    # wait until the close ramp finishes (recommended)
    wait_for_close_ramp: bool = True,
    close_ramp_buffer_name: str = "_hand_close_ramp",
    close_ramp_steps: int = 30,
    # do it once per episode
    once: bool = True,
    switched_buffer_name: str = "_goal_switched",
) -> None:
    """After the hand closes, override the commanded goal position (x,y,z) in base frame.

    Keeps the commanded orientation (qw,qx,qy,qz) unchanged.
    """
    if not hasattr(env, latch_buffer_name):
        return

    hand_closed_all = getattr(env, latch_buffer_name)
    mask = hand_closed_all[env_ids].clone()

    if wait_for_close_ramp:
        if not hasattr(env, close_ramp_buffer_name):
            return
        ramp = getattr(env, close_ramp_buffer_name)
        mask &= (ramp[env_ids] >= int(close_ramp_steps))

    if not mask.any():
        return

    if once:
        if not hasattr(env, switched_buffer_name):
            setattr(env, switched_buffer_name, torch.zeros(env.num_envs, device=env.device, dtype=torch.bool))
        switched = getattr(env, switched_buffer_name)
        mask &= ~switched[env_ids]
        if not mask.any():
            return

    ids = env_ids[mask.nonzero(as_tuple=False).squeeze(-1)]

    cmd = env.command_manager.get_command(command_name).clone()  # (N,7)
    pos = torch.tensor(target_pos, device=env.device, dtype=cmd.dtype).view(1, 3)
    cmd[ids, 0:3] = pos

    env.command_manager.set_command(command_name, cmd)

    if once:
        switched[ids] = True


def open_hand_once_when_at_current_goal(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    ee_cfg: SceneEntityCfg,
    hand_joint_cfg: SceneEntityCfg,
    command_name: str,
    # Only open after goal has been switched (i.e., we’re on goal #2)
    goal_switched_buffer_name: str = "_goal_switched",
    reach_threshold: float = 0.05,
    # latch open so it stays open after triggering
    opened_buffer_name: str = "_hand_opened_drop",
    # smooth opening ramp
    open_ramp_steps: int = 20,
    ramp_buffer_name: str = "_hand_open_ramp",
    start_buffer_name: str = "_hand_open_start",
    # joint targets
    open_pos=None,
    open_pos_is_degrees: bool = True,
) -> None:
    """After the goal switches (goal #2 active), open the hand once when EE reaches the current goal."""

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


def reset_hand_open_start_buffer(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    hand_joint_cfg: SceneEntityCfg,
    buffer_name: str = "_hand_open_start",
) -> None:
    """Ensure _hand_open_start is (num_envs, J) and reset rows for env_ids."""
    robot: Articulation = env.scene[hand_joint_cfg.name]
    joint_names = getattr(hand_joint_cfg, "joint_names", None)
    if not joint_names:
        raise ValueError("hand_joint_cfg.joint_names must be provided for reset_hand_open_start_buffer.")

    J = len(joint_names)
    dtype = robot.data.default_joint_pos.dtype

    need_new = True
    if hasattr(env, buffer_name):
        buf = getattr(env, buffer_name)
        if isinstance(buf, torch.Tensor) and buf.ndim == 2 and buf.shape[0] == env.num_envs and buf.shape[1] == J:
            need_new = False

    if need_new:
        setattr(env, buffer_name, torch.zeros((env.num_envs, J), device=env.device, dtype=dtype))

    buf = getattr(env, buffer_name)
    buf[env_ids] = 0.0


def reset_open_hand_after_second_goal_buffers_on_episode_start(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    hand_joint_cfg: SceneEntityCfg,
    goal_switched_buffer_name: str = "_goal_switched",
    opened_buffer_name: str = "_hand_opened_drop",
    ramp_buffer_name: str = "_hand_open_ramp",
    start_buffer_name: str = "_hand_open_start",
) -> None:
    if hasattr(env, "reset_buf"):
        just_reset = env.reset_buf[env_ids].to(torch.bool)
    elif hasattr(env, "episode_length_buf"):
        just_reset = (env.episode_length_buf[env_ids] == 0)
    else:
        return

    if not just_reset.any():
        return

    ids = env_ids[just_reset.nonzero(as_tuple=False).squeeze(-1)]

    reset_named_buffer(env, ids, buffer_name=goal_switched_buffer_name, dtype="bool")
    reset_named_buffer(env, ids, buffer_name=opened_buffer_name, dtype="bool")
    reset_named_buffer(env, ids, buffer_name=ramp_buffer_name, dtype="int")
    reset_hand_open_start_buffer(env, ids, hand_joint_cfg=hand_joint_cfg, buffer_name=start_buffer_name)


from isaaclab.envs import ManagerBasedRLEnv


def switch_goal_position_after_hand_close(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    command_name: str,
    target_pos=(0.3, 0.05, 0.25),
    # Only AFTER the hand is actually closed
    latch_buffer_name: str = "_hand_closed",
    wait_for_close_ramp: bool = True,
    close_ramp_buffer_name: str = "_hand_close_ramp",
    close_ramp_steps: int = 30,
    # Switch once per episode
    once: bool = True,
    switched_buffer_name: str = "_goal_switched",
) -> None:
    """After the hand closes, override the commanded goal XYZ (in base/root frame).

    Keeps the commanded orientation (qw,qx,qy,qz) unchanged if present.
    """
    # Need the hand-closed latch
    if not hasattr(env, latch_buffer_name):
        return
    hand_closed = getattr(env, latch_buffer_name)
    mask = hand_closed[env_ids].clone()

    # Optionally wait until the close ramp finishes
    if wait_for_close_ramp:
        if not hasattr(env, close_ramp_buffer_name):
            return
        ramp = getattr(env, close_ramp_buffer_name)
        mask &= (ramp[env_ids] >= int(close_ramp_steps))

    if not mask.any():
        return

    # Optional "once per episode" latch
    if once:
        if not hasattr(env, switched_buffer_name):
            setattr(env, switched_buffer_name, torch.zeros((env.num_envs,), device=env.device, dtype=torch.bool))
        switched = getattr(env, switched_buffer_name)
        mask &= ~switched[env_ids]
        if not mask.any():
            return

    ids = env_ids[mask.nonzero(as_tuple=False).squeeze(-1)]

    # Get current command buffer
    cmd = env.command_manager.get_command(command_name)
    if cmd.shape[-1] < 3:
        raise ValueError(f"Command '{command_name}' must have at least xyz. Got shape {tuple(cmd.shape)}")

    # Set xyz to target_pos (leave the rest untouched)
    tgt = torch.tensor(target_pos, device=cmd.device, dtype=cmd.dtype).view(1, 3).repeat(ids.numel(), 1)

    if hasattr(env.command_manager, "set_command"):
        cmd_new = cmd.clone()
        cmd_new[ids, 0:3] = tgt
        env.command_manager.set_command(command_name, cmd_new)
    else:
        # Many IsaacLab builds return a writable tensor here, so in-place works
        cmd[ids, 0:3] = tgt

    if once:
        switched[ids] = True


import torch
from isaaclab.assets import Articulation
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import SceneEntityCfg


def open_hand_once_when_at_current_goal(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    ee_cfg: SceneEntityCfg,
    hand_joint_cfg: SceneEntityCfg,
    command_name: str,
    goal_switched_buffer_name: str = "_goal_switched",
    reach_threshold: float = 0.06,
    opened_buffer_name: str = "_hand_opened_drop",
    open_ramp_steps: int = 20,
    ramp_buffer_name: str = "_hand_open_ramp",
    start_buffer_name: str = "_hand_open_start",
    open_pos=None,
    open_pos_is_degrees: bool = True,
) -> None:
    """After goal switches (goal #2), open the hand once when EE reaches the current goal."""

    # Must be on goal #2
    if not hasattr(env, goal_switched_buffer_name):
        return
    goal_switched = getattr(env, goal_switched_buffer_name)
    switched_mask = goal_switched[env_ids]
    if not switched_mask.any():
        return

    # Check reach to current goal
    goal_pos = env.command_manager.get_command(command_name)[:, 0:3]
    ee_pos = body_pos_in_root_frame(env, ee_cfg)
    dist = torch.linalg.norm(ee_pos[env_ids] - goal_pos[env_ids], dim=-1)
    reached_mask = dist < reach_threshold

    can_open = switched_mask & reached_mask
    # if nobody can open and we don't even have opened buffer yet, nothing to do
    if not can_open.any() and not hasattr(env, opened_buffer_name):
        return

    robot: Articulation = env.scene[hand_joint_cfg.name]

    joint_names = getattr(hand_joint_cfg, "joint_names", None)
    if not joint_names:
        raise ValueError("hand_joint_cfg.joint_names must be provided for open_hand_once_when_at_current_goal.")

    # name -> id map (cache)
    if not hasattr(env, "_joint_name_to_id"):
        names_all = list(robot.data.joint_names)
        env._joint_name_to_id = {n: i for i, n in enumerate(names_all)}
    name_to_id = env._joint_name_to_id

    joint_ids = torch.tensor([int(name_to_id[n]) for n in joint_names], device=env.device, dtype=torch.int64)
    J = int(joint_ids.numel())
    dtype = robot.data.default_joint_pos.dtype

    # Spec -> vector (J,)
    def _spec_to_vec(spec, default_val: float = 0.0) -> torch.Tensor:
        if spec is None:
            return torch.full((J,), float(default_val), device=env.device, dtype=dtype)
        if isinstance(spec, (float, int)):
            return torch.full((J,), float(spec), device=env.device, dtype=dtype)
        if isinstance(spec, dict):
            v = torch.full((J,), float(default_val), device=env.device, dtype=dtype)
            for j, n in enumerate(joint_names):
                if n in spec:
                    v[j] = float(spec[n])
            return v
        if isinstance(spec, (list, tuple)):
            if len(spec) != J:
                raise ValueError(f"open_pos length mismatch: got {len(spec)}, expected {J}.")
            return torch.tensor(spec, device=env.device, dtype=dtype)
        raise TypeError(f"Unsupported open_pos type: {type(spec)}")

    open_vec = _spec_to_vec(open_pos, default_val=0.0)
    if open_pos_is_degrees:
        open_vec = torch.deg2rad(open_vec)

    # Buffers
    if not hasattr(env, opened_buffer_name):
        setattr(env, opened_buffer_name, torch.zeros((env.num_envs,), device=env.device, dtype=torch.bool))
    opened = getattr(env, opened_buffer_name)

    if not hasattr(env, ramp_buffer_name):
        setattr(env, ramp_buffer_name, torch.zeros((env.num_envs,), device=env.device, dtype=torch.int32))
    ramp_ctr = getattr(env, ramp_buffer_name)

    if not hasattr(env, start_buffer_name):
        setattr(env, start_buffer_name, torch.zeros((env.num_envs, J), device=env.device, dtype=dtype))
    start_buf = getattr(env, start_buffer_name)

    # Newly opened envs: capture start pose
    newly = can_open & (~opened[env_ids])
    if newly.any():
        new_ids = env_ids[newly.nonzero(as_tuple=False).squeeze(-1)]
        start_buf[new_ids] = robot.data.joint_pos[new_ids][:, joint_ids]
        ramp_ctr[new_ids] = 0
        opened[new_ids] = True

    # Apply opening to all opened envs (keeps them open)
    opened_mask = opened[env_ids]
    if not opened_mask.any():
        return
    use_ids = env_ids[opened_mask.nonzero(as_tuple=False).squeeze(-1)]

    # Ramp
    if open_ramp_steps is None or int(open_ramp_steps) <= 0:
        alpha = torch.ones((use_ids.numel(), 1), device=env.device, dtype=dtype)
    else:
        ramp_ctr[use_ids] = torch.clamp(ramp_ctr[use_ids] + 1, max=int(open_ramp_steps))
        alpha = (ramp_ctr[use_ids].to(dtype) / float(open_ramp_steps)).clamp(0.0, 1.0).view(-1, 1)

    start_pos = start_buf[use_ids]  # (B,J)
    goal_pos = open_vec.view(1, -1).repeat(use_ids.numel(), 1)
    targets = start_pos + alpha * (goal_pos - start_pos)

    # Clamp if limits exist
    if hasattr(robot.data, "joint_pos_limits") and robot.data.joint_pos_limits is not None:
        lim = robot.data.joint_pos_limits[use_ids][:, joint_ids, :]  # (B,J,2)
        targets = torch.clamp(targets, lim[..., 0], lim[..., 1])

    robot.set_joint_position_target(targets, joint_ids=joint_ids, env_ids=use_ids)


def reset_hand_open_start_buffer(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    hand_joint_cfg: SceneEntityCfg,
    buffer_name: str = "_hand_open_start",
) -> None:
    """Ensure _hand_open_start is (num_envs, J) and reset rows for env_ids."""
    robot: Articulation = env.scene[hand_joint_cfg.name]
    joint_names = getattr(hand_joint_cfg, "joint_names", None)
    if not joint_names:
        raise ValueError("hand_joint_cfg.joint_names must be provided for reset_hand_open_start_buffer.")

    J = len(joint_names)
    dtype = robot.data.default_joint_pos.dtype

    need_new = True
    if hasattr(env, buffer_name):
        buf = getattr(env, buffer_name)
        if isinstance(buf, torch.Tensor) and buf.ndim == 2 and buf.shape == (env.num_envs, J):
            need_new = False

    if need_new:
        setattr(env, buffer_name, torch.zeros((env.num_envs, J), device=env.device, dtype=dtype))

    getattr(env, buffer_name)[env_ids] = 0.0


def reset_open_hand_after_second_goal_buffers_on_episode_start(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    hand_joint_cfg: SceneEntityCfg,
    goal_switched_buffer_name: str = "_goal_switched",
    opened_buffer_name: str = "_hand_opened_drop",
    ramp_buffer_name: str = "_hand_open_ramp",
    start_buffer_name: str = "_hand_open_start",
) -> None:
    """Hard re-arm the open-after-goal2 pipeline right when an env starts a new episode."""
    # Detect "just reset" envs
    if hasattr(env, "reset_buf"):
        just_reset = env.reset_buf[env_ids].to(torch.bool)
    elif hasattr(env, "episode_length_buf"):
        just_reset = (env.episode_length_buf[env_ids] == 0)
    else:
        return

    if not just_reset.any():
        return

    ids = env_ids[just_reset.nonzero(as_tuple=False).squeeze(-1)]

    reset_named_buffer(env, ids, buffer_name=goal_switched_buffer_name, dtype="bool")
    reset_named_buffer(env, ids, buffer_name=opened_buffer_name, dtype="bool")
    reset_named_buffer(env, ids, buffer_name=ramp_buffer_name, dtype="int")
    reset_hand_open_start_buffer(env, ids, hand_joint_cfg=hand_joint_cfg, buffer_name=start_buffer_name)
