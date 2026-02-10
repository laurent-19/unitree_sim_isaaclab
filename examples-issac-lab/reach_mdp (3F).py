# reach_mdp.py
# Minimal custom MDP helpers for the G1 reach task (task-space reaching).

from __future__ import annotations

import torch

from isaaclab.assets import Articulation
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import subtract_frame_transforms
from isaaclab.utils.math import quat_apply

def _resolve_single_body_id(asset: Articulation, asset_cfg: SceneEntityCfg) -> int:
    """Resolve a single body id from a SceneEntityCfg.

    In some manager paths (notably EventTerm params), SceneEntityCfg may not be pre-resolved,
    so `body_ids` can be a `slice` sentinel. In that case, we resolve using `body_names`.
    """
    body_ids = getattr(asset_cfg, "body_ids", None)

    # Common case: already resolved to an indexable container / tensor
    if body_ids is not None and not isinstance(body_ids, slice):
        return int(body_ids[0])

    # Fallback: resolve from the provided body name(s)
    body_names = getattr(asset_cfg, "body_names", None)
    if not body_names or len(body_names) != 1:
        raise ValueError(
            "Expected SceneEntityCfg with exactly one body (body_names=[...]) "
            "or resolved body_ids=[...]. Got body_names=%s, body_ids=%s" % (body_names, body_ids)
        )

    name0 = body_names[0]

    # IsaacLab Articulation usually provides find_bodies; fall back to name lookup if needed.
    if hasattr(asset, "find_bodies"):
        res = asset.find_bodies([name0])
        ids = res[0] if isinstance(res, tuple) else res
        return int(ids[0])

    if hasattr(asset, "data") and hasattr(asset.data, "body_names"):
        return int(list(asset.data.body_names).index(name0))

    raise AttributeError("Cannot resolve body id: Articulation has no find_bodies and no data.body_names.")



def body_pos_in_root_frame(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Return body position expressed in the asset's root frame.

    This matches how UniformPoseCommand samples goals (in base/root frame), so the policy
    sees current EE position and goal in the same frame.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    body_id = _resolve_single_body_id(asset, asset_cfg)

    root_pos_w = asset.data.root_pos_w
    root_quat_w = asset.data.root_quat_w

    body_pos_w = asset.data.body_pos_w[:, body_id]
    body_quat_w = asset.data.body_quat_w[:, body_id]

    pos_b, _ = subtract_frame_transforms(root_pos_w, root_quat_w, body_pos_w, body_quat_w)
    return pos_b


def reached_ee_position(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    command_name: str,
    threshold: float = 0.05,
) -> torch.Tensor:
    """Terminate when the end-effector is within `threshold` meters of the commanded goal (in base frame)."""
    goal = env.command_manager.get_command(command_name)[:, 0:3]  # (x,y,z) in base frame
    ee_pos_b = body_pos_in_root_frame(env, asset_cfg)
    return torch.linalg.norm(ee_pos_b - goal, dim=-1) < threshold


def hold_joints_at_default_targets(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> None:
    """Hold specified joints at their default targets (does NOT reset the whole robot state)."""
    asset: Articulation = env.scene[asset_cfg.name]

    # default targets for the selected joints
    default_pos = asset.data.default_joint_pos[env_ids][:, asset_cfg.joint_ids]
    default_vel = asset.data.default_joint_vel[env_ids][:, asset_cfg.joint_ids]

    asset.set_joint_position_target(default_pos, joint_ids=asset_cfg.joint_ids, env_ids=env_ids)
    asset.set_joint_velocity_target(default_vel, joint_ids=asset_cfg.joint_ids, env_ids=env_ids)


def position_command_error(
    env: ManagerBasedRLEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """L2 dista:contentReference[oaicite:4]{index=4}on and commanded goal (both in robot root/base frame)."""
    goal_b = env.command_manager.get_command(command_name)[:, :3]  # (x,y,z) in base/root frame
    ee_pos_b = body_pos_in_root_frame(env, asset_cfg)
    return torch.linalg.norm(ee_pos_b - goal_b, dim=-1)


def position_command_error_tanh(
    env: ManagerBasedRLEnv,
    std: float,
    command_name: str,
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Dense reward: 1 - tanh(distance/std). Higher is better."""
    dist = position_command_error(env, command_name=command_name, asset_cfg=asset_cfg)
    return 1.0 - torch.tanh(dist / std)


def body_quat_in_root_frame(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    body_id = _resolve_single_body_id(asset, asset_cfg)

    root_pos_w = asset.data.root_pos_w
    root_quat_w = asset.data.root_quat_w
    body_pos_w = asset.data.body_pos_w[:, body_id]
    body_quat_w = asset.data.body_quat_w[:, body_id]

    _, quat_b = subtract_frame_transforms(root_pos_w, root_quat_w, body_pos_w, body_quat_w)
    return quat_b  # (w, x, y, z)


def orientation_command_error(env: ManagerBasedRLEnv, command_name: str, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    goal = env.command_manager.get_command(command_name)[:, 3:7]  # qw,qx,qy,qz in root frame
    cur = body_quat_in_root_frame(env, asset_cfg)

    # quaternion angular distance
    dot = torch.abs(torch.sum(cur * goal, dim=-1))
    dot = torch.clamp(dot, -1.0, 1.0)
    return 2.0 * torch.acos(dot)  # radians


def orientation_command_error_tanh(env: ManagerBasedRLEnv, std: float, command_name: str, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    ang = orientation_command_error(env, command_name=command_name, asset_cfg=asset_cfg)
    return 1.0 - torch.tanh(ang / std)


def reached_ee_pose(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    command_name: str,
    pos_threshold: float = 0.05,
    rot_threshold: float = 0.35,
) -> torch.Tensor:
    pos_ok = reached_ee_position(env, asset_cfg=asset_cfg, command_name=command_name, threshold=pos_threshold)
    rot_ok = orientation_command_error(env, command_name=command_name, asset_cfg=asset_cfg) < rot_threshold
    return pos_ok & rot_ok


def body_quat_in_root_frame(env, asset_cfg):
    """Return body orientation expressed in the asset's root frame (wxyz)."""
    asset = env.scene[asset_cfg.name]
    body_id = _resolve_single_body_id(asset, asset_cfg)

    root_pos_w = asset.data.root_pos_w
    root_quat_w = asset.data.root_quat_w

    body_pos_w = asset.data.body_pos_w[:, body_id]
    body_quat_w = asset.data.body_quat_w[:, body_id]

    _, quat_b = subtract_frame_transforms(root_pos_w, root_quat_w, body_pos_w, body_quat_w)
    return quat_b


def position_command_error(env, asset_cfg, command_name: str):
    """L2 position error in the robot/root frame."""
    goal_pos = env.command_manager.get_command(command_name)[:, 0:3]
    ee_pos = body_pos_in_root_frame(env, asset_cfg)
    return torch.linalg.norm(ee_pos - goal_pos, dim=-1)


def position_command_error_tanh(env, asset_cfg, command_name: str, std: float = 0.15):
    """Smooth bounded tracking reward in [0,1]."""
    err = position_command_error(env, asset_cfg, command_name)
    return 1.0 - torch.tanh(err / std)


# ----------------------------
# Quaternion vector rotation (wxyz)
# ----------------------------

def _quat_rotate_wxyz(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """
    Rotate vector(s) v by quaternion(s) q. q is (...,4) in wxyz, v is (...,3).
    """
    w = q[..., 0:1]
    xyz = q[..., 1:4]
    t = 2.0 * torch.cross(xyz, v, dim=-1)
    return v + w * t + torch.cross(xyz, t, dim=-1)


# ----------------------------
# 5-DOF-friendly "orientation" reward:
# align ONE axis (e.g., palm normal) to ONE goal axis (e.g., goal z-axis)
# ----------------------------

def palm_axis_align_to_goal_tanh(
    env,
    asset_cfg,
    command_name: str,
    palm_axis=(0.0, 0.0, 1.0),   # axis in the palm link's LOCAL frame
    goal_axis=(0.0, 0.0, 1.0),   # axis in the goal frame's LOCAL frame
    pos_std: float = 0.12,       # gating: only care about orientation when close
    ori_std: float = 0.8,        # softness for axis mismatch
):
    """
    Returns a bounded reward in [0,1] that:
      - gates on positional closeness
      - encourages alignment of palm_axis (in EE frame) with goal_axis (in goal frame)
    This is the right target for 5DOF arms (direction alignment, not full roll).
    """
    # gate by position (so we don't do goofy far-away orientation chasing)
    pos_gate = position_command_error_tanh(env, asset_cfg, command_name, std=pos_std)

    ee_q = body_quat_in_root_frame(env, asset_cfg)                 # (N,4) wxyz
    goal_q = env.command_manager.get_command(command_name)[:, 3:7] # (N,4) wxyz

    device = goal_q.device
    palm_axis_v = torch.tensor(palm_axis, device=device, dtype=goal_q.dtype).view(1, 3).expand(goal_q.shape[0], 3)
    goal_axis_v = torch.tensor(goal_axis, device=device, dtype=goal_q.dtype).view(1, 3).expand(goal_q.shape[0], 3)

    ee_dir = _quat_rotate_wxyz(ee_q, palm_axis_v)
    goal_dir = _quat_rotate_wxyz(goal_q, goal_axis_v)

    # cosine alignment
    cos = torch.sum(ee_dir * goal_dir, dim=-1)
    cos = torch.clamp(cos, -1.0, 1.0)

    # error in [0,2], 0 is perfect
    err = 1.0 - cos
    ori_rew = 1.0 - torch.tanh(err / ori_std)

    return pos_gate * ori_rew



def body_quat_in_root_frame(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    body_id = _resolve_single_body_id(asset, asset_cfg)

    root_pos_w = asset.data.root_pos_w
    root_quat_w = asset.data.root_quat_w

    body_pos_w = asset.data.body_pos_w[:, body_id]
    body_quat_w = asset.data.body_quat_w[:, body_id]

    _, quat_b = subtract_frame_transforms(root_pos_w, root_quat_w, body_pos_w, body_quat_w)
    return quat_b  # (w,x,y,z) in root frame


def _normalize_quat(q: torch.Tensor) -> torch.Tensor:
    return q / (torch.linalg.norm(q, dim=-1, keepdim=True) + 1e-8)

def orientation_command_error(env: ManagerBasedRLEnv, command_name: str, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    goal = env.command_manager.get_command(command_name)[:, 3:7]  # qw,qx,qy,qz
    cur = body_quat_in_root_frame(env, asset_cfg)

    goal = _normalize_quat(goal)
    cur = _normalize_quat(cur)

    dot = torch.abs(torch.sum(cur * goal, dim=-1))
    dot = torch.nan_to_num(dot, nan=0.0, posinf=1.0, neginf=0.0)
    dot = torch.clamp(dot, 0.0, 1.0)

    return 2.0 * torch.acos(dot)

def _sensor_max_force_norm(env, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """Max contact force norm per-env for a ContactSensor."""
    sensor = env.scene.sensors[sensor_cfg.name]
    forces = sensor.data.net_forces_w  # typically (N, M, 3) or (N, 1, 3)

    # reshape to (N, K, 3) robustly
    N = forces.shape[0]
    forces = forces.reshape(N, -1, 3)

    # norms: (N, K)
    norms = torch.linalg.norm(forces, dim=-1)
    norms = torch.nan_to_num(norms, nan=0.0, posinf=0.0, neginf=0.0)

    # max over contacts: (N,)
    max_norm = norms.max(dim=1).values
    return max_norm


def contact_force_reward(
    env,
    sensor_cfg: SceneEntityCfg,
    min_force: float = 1.0,
    max_force: float = 50.0,
) -> torch.Tensor:
    """
    Returns a bounded value in [0, 1] based on the sensor's max contact force norm:
      0 if <= min_force
      1 if >= max_force
      linear in between
    Perfect to use as a penalty by giving it a negative weight.
    """
    f = _sensor_max_force_norm(env, sensor_cfg)
    denom = max(max_force - min_force, 1e-6)
    x = (f - min_force) / denom
    return torch.clamp(x, 0.0, 1.0)


def contact_force_exceeds(env, sensor_cfg, force_thresh: float):
    sensor = env.scene.sensors[sensor_cfg.name]
    # robust norm: works whether it's (N,3) or (N,1,3)
    f = torch.linalg.norm(sensor.data.net_forces_w.reshape(sensor.data.net_forces_w.shape[0], -1), dim=-1)
    return f > force_thresh


def orientation_command_error_tanh_gated(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    command_name: str,
    std: float = 0.6,
    pos_gate: float = 0.12,
) -> torch.Tensor:
    # gate orientation reward by how close we are in position (prevents fighting early)
    goal_pos = env.command_manager.get_command(command_name)[:, 0:3]
    ee_pos = body_pos_in_root_frame(env, asset_cfg)
    pos_err = torch.linalg.norm(ee_pos - goal_pos, dim=-1)

    ang_err = orientation_command_error(env, asset_cfg, command_name)
    rot_rew = 1.0 - torch.tanh(ang_err / std)

    gate = (pos_err < pos_gate).to(rot_rew.dtype)
    return gate * rot_rew


def reached_ee_pose(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    command_name: str,
    pos_threshold: float = 0.05,
    rot_threshold: float = 0.35,
) -> torch.Tensor:
    goal = env.command_manager.get_command(command_name)
    goal_pos = goal[:, 0:3]

    ee_pos = body_pos_in_root_frame(env, asset_cfg)
    pos_ok = torch.linalg.norm(ee_pos - goal_pos, dim=-1) < pos_threshold

    ang_err = orientation_command_error(env, asset_cfg, command_name)
    rot_ok = ang_err < rot_threshold

    return pos_ok & rot_ok


import torch

def _quat_rotate_wxyz(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    # q: (N,4) wxyz, v: (N,3)
    w = q[:, 0:1]
    xyz = q[:, 1:4]
    t = 2.0 * torch.cross(xyz, v, dim=-1)
    return v + w * t + torch.cross(xyz, t, dim=-1)

def axis_alignment_cos(
    env,
    asset_cfg,
    command_name: str,
    palm_axis=(0.0, 0.0, 1.0),
    goal_axis=(0.0, 0.0, 1.0),
):
    """Cosine alignment between a palm local axis and a goal-frame local axis."""
    ee_q = body_quat_in_root_frame(env, asset_cfg)                 # (N,4) wxyz in root frame
    goal_q = env.command_manager.get_command(command_name)[:, 3:7] # (N,4) wxyz in root frame

    device = goal_q.device
    dtype = goal_q.dtype

    pa = torch.tensor(palm_axis, device=device, dtype=dtype).view(1, 3).repeat(goal_q.shape[0], 1)
    ga = torch.tensor(goal_axis, device=device, dtype=dtype).view(1, 3).repeat(goal_q.shape[0], 1)

    ee_dir = _quat_rotate_wxyz(ee_q, pa)
    goal_dir = _quat_rotate_wxyz(goal_q, ga)

    ee_dir = ee_dir / (torch.linalg.norm(ee_dir, dim=-1, keepdim=True) + 1e-8)
    goal_dir = goal_dir / (torch.linalg.norm(goal_dir, dim=-1, keepdim=True) + 1e-8)

    cos = torch.sum(ee_dir * goal_dir, dim=-1)
    return torch.clamp(cos, -1.0, 1.0)

def axis_align_reward_gated(
    env,
    asset_cfg,
    command_name: str,
    palm_axis=(0.0, 0.0, 1.0),
    goal_axis=(0.0, 0.0, 1.0),
    pos_gate: float = 0.12,
    sharpness: float = 6.0,
):
    """
    Reward in [0,1]. Only turns on near the target position, then strongly encourages axis alignment.
    """
    # gate by distance to target (root frame)
    goal_pos = env.command_manager.get_command(command_name)[:, 0:3]
    ee_pos = body_pos_in_root_frame(env, asset_cfg)
    pos_err = torch.linalg.norm(ee_pos - goal_pos, dim=-1)

    gate = (pos_err < pos_gate).to(ee_pos.dtype)

    cos = axis_alignment_cos(env, asset_cfg, command_name, palm_axis=palm_axis, goal_axis=goal_axis)
    # map cos -> [0,1] with strong slope near 1
    align = torch.clamp((cos + 1.0) * 0.5, 0.0, 1.0)
    align = align ** sharpness
    return gate * align

def reached_pos_and_axis(
    env,
    asset_cfg,
    command_name: str,
    pos_threshold: float = 0.05,
    cos_threshold: float = 0.97,   # ~14 degrees
    palm_axis=(0.0, 0.0, 1.0),
    goal_axis=(0.0, 0.0, 1.0),
):
    goal_pos = env.command_manager.get_command(command_name)[:, 0:3]
    ee_pos = body_pos_in_root_frame(env, asset_cfg)
    pos_ok = torch.linalg.norm(ee_pos - goal_pos, dim=-1) < pos_threshold

    cos = axis_alignment_cos(env, asset_cfg, command_name, palm_axis=palm_axis, goal_axis=goal_axis)
    rot_ok = cos > cos_threshold
    return pos_ok & rot_ok


def orientation_bonus_softgated(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    command_name: str,
    ori_std: float = 0.6,     # ~34 deg softness
    pos_std: float = 0.12,    # gate turns on as we get close
) -> torch.Tensor:
    """
    Bonus reward in [0,1]:
      - 0 far away (doesn't interfere with reaching)
      - increases near the target
      - rewards quaternion closeness as much as the 5DOF arm can achieve
    """
    # smooth "near-target" gate (0..1)
    pos_gate = position_command_error_tanh(env, asset_cfg=asset_cfg, command_name=command_name, std=pos_std)

    # smooth orientation bonus (0..1)
    ang = orientation_command_error(env, asset_cfg=asset_cfg, command_name=command_name)
    ori_bonus = 1.0 - torch.tanh(ang / ori_std)

    return pos_gate * ori_bonus


# ============================
# SAFE OVERRIDES (keep last)
# ============================

def _normalize_quat(q: torch.Tensor) -> torch.Tensor:
    return q / (torch.linalg.norm(q, dim=-1, keepdim=True) + 1e-8)


def orientation_command_error(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    command_name: str,
) -> torch.Tensor:
    """Quaternion angular distance (radians) between EE and goal (both in root frame)."""
    goal = env.command_manager.get_command(command_name)[:, 3:7]  # qw,qx,qy,qz
    cur = body_quat_in_root_frame(env, asset_cfg)

    goal = _normalize_quat(goal)
    cur = _normalize_quat(cur)

    dot = torch.abs(torch.sum(cur * goal, dim=-1))
    dot = torch.nan_to_num(dot, nan=0.0, posinf=1.0, neginf=0.0)
    dot = torch.clamp(dot, 0.0, 1.0)

    ang = 2.0 * torch.acos(dot)
    return torch.nan_to_num(ang, nan=0.0, posinf=0.0, neginf=0.0)


def orientation_command_error_tanh_gated(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    command_name: str,
    std: float = 0.6,
    pos_gate: float = 0.12,
) -> torch.Tensor:
    goal_pos = env.command_manager.get_command(command_name)[:, 0:3]
    ee_pos = body_pos_in_root_frame(env, asset_cfg)
    pos_err = torch.linalg.norm(ee_pos - goal_pos, dim=-1)

    ang_err = orientation_command_error(env, asset_cfg=asset_cfg, command_name=command_name)
    rot_rew = 1.0 - torch.tanh(ang_err / std)

    gate = (pos_err < pos_gate).to(rot_rew.dtype)
    out = gate * rot_rew
    return torch.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)


def reached_ee_pose(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    command_name: str,
    pos_threshold: float = 0.05,
    rot_threshold: float = 0.35,
) -> torch.Tensor:
    goal_pos = env.command_manager.get_command(command_name)[:, 0:3]
    ee_pos = body_pos_in_root_frame(env, asset_cfg)
    pos_ok = torch.linalg.norm(ee_pos - goal_pos, dim=-1) < pos_threshold

    ang_err = orientation_command_error(env, asset_cfg=asset_cfg, command_name=command_name)
    rot_ok = ang_err < rot_threshold

    return pos_ok & rot_ok


def orientation_bonus_softgated(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    command_name: str,
    ori_std: float = 0.6,
    pos_std: float = 0.12,
) -> torch.Tensor:
    pos_gate = position_command_error_tanh(env, asset_cfg=asset_cfg, command_name=command_name, std=pos_std)
    ang = orientation_command_error(env, asset_cfg=asset_cfg, command_name=command_name)
    ori_bonus = 1.0 - torch.tanh(ang / ori_std)

    out = pos_gate * ori_bonus
    return torch.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)


from isaaclab.utils.math import quat_apply

def _resolve_single_body_id(asset: Articulation, asset_cfg: SceneEntityCfg) -> int:
    """Resolve a single body id from a SceneEntityCfg.

    In some manager paths (notably EventTerm params), SceneEntityCfg may not be pre-resolved,
    so `body_ids` can be a `slice` sentinel. In that case, we resolve using `body_names`.
    """
    body_ids = getattr(asset_cfg, "body_ids", None)

    # Common case: already resolved to an indexable container / tensor
    if body_ids is not None and not isinstance(body_ids, slice):
        return int(body_ids[0])

    # Fallback: resolve from the provided body name(s)
    body_names = getattr(asset_cfg, "body_names", None)
    if not body_names or len(body_names) != 1:
        raise ValueError(
            "Expected SceneEntityCfg with exactly one body (body_names=[...]) "
            "or resolved body_ids=[...]. Got body_names=%s, body_ids=%s" % (body_names, body_ids)
        )

    name0 = body_names[0]

    # IsaacLab Articulation usually provides find_bodies; fall back to name lookup if needed.
    if hasattr(asset, "find_bodies"):
        res = asset.find_bodies([name0])
        ids = res[0] if isinstance(res, tuple) else res
        return int(ids[0])

    if hasattr(asset, "data") and hasattr(asset.data, "body_names"):
        return int(list(asset.data.body_names).index(name0))

    raise AttributeError("Cannot resolve body id: Articulation has no find_bodies and no data.body_names.")


def place_object_near_ee_goal(
    env,
    env_ids: torch.Tensor,
    object_cfg: SceneEntityCfg,
    robot_cfg: SceneEntityCfg,
    command_name: str,
    offset_root=(0.0, -0.02, -0.04),
):
    obj = env.scene[object_cfg.name]
    robot = env.scene[robot_cfg.name]

    # Command is in robot root frame: [x,y,z, qw,qx,qy,qz]
    cmd = env.command_manager.get_command(command_name)[env_ids]
    goal_pos_root = cmd[:, 0:3]

    root_pos_w = robot.data.root_pos_w[env_ids]
    root_quat_w = robot.data.root_quat_w[env_ids]

    # root->world
    goal_pos_w = root_pos_w + quat_apply(root_quat_w, goal_pos_root)

    # offset (also in root frame)
    off = torch.tensor(offset_root, device=env.device, dtype=goal_pos_root.dtype).view(1, 3).repeat(env_ids.numel(), 1)
    off_w = quat_apply(root_quat_w, off)

    obj_pos_w = goal_pos_w + off_w
    obj_quat_w = torch.tensor([1.0, 0.0, 0.0, 0.0], device=env.device, dtype=goal_pos_root.dtype).view(1, 4).repeat(env_ids.numel(), 1)

    obj.write_root_pose_to_sim(torch.cat([obj_pos_w, obj_quat_w], dim=-1), env_ids=env_ids)
    if not hasattr(env, "_cyl_spawn_pos_w"):
        env._cyl_spawn_pos_w = torch.zeros((env.num_envs, 3), device=env.device, dtype=obj_pos_w.dtype)

    env._cyl_spawn_pos_w[env_ids] = obj_pos_w
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
    """Place object near the current EE goal exactly once per episode (first step after reset)."""
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


from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import quat_apply

def _resolve_single_body_id(asset: Articulation, asset_cfg: SceneEntityCfg) -> int:
    """Resolve a single body id from a SceneEntityCfg.

    In some manager paths (notably EventTerm params), SceneEntityCfg may not be pre-resolved,
    so `body_ids` can be a `slice` sentinel. In that case, we resolve using `body_names`.
    """
    body_ids = getattr(asset_cfg, "body_ids", None)

    # Common case: already resolved to an indexable container / tensor
    if body_ids is not None and not isinstance(body_ids, slice):
        return int(body_ids[0])

    # Fallback: resolve from the provided body name(s)
    body_names = getattr(asset_cfg, "body_names", None)
    if not body_names or len(body_names) != 1:
        raise ValueError(
            "Expected SceneEntityCfg with exactly one body (body_names=[...]) "
            "or resolved body_ids=[...]. Got body_names=%s, body_ids=%s" % (body_names, body_ids)
        )

    name0 = body_names[0]

    # IsaacLab Articulation usually provides find_bodies; fall back to name lookup if needed.
    if hasattr(asset, "find_bodies"):
        res = asset.find_bodies([name0])
        ids = res[0] if isinstance(res, tuple) else res
        return int(ids[0])

    if hasattr(asset, "data") and hasattr(asset.data, "body_names"):
        return int(list(asset.data.body_names).index(name0))

    raise AttributeError("Cannot resolve body id: Articulation has no find_bodies and no data.body_names.")


def object_upright_tilt_error(env, object_cfg: SceneEntityCfg) -> torch.Tensor:
    """0 when perfectly upright, grows as object tips over."""
    obj = env.scene[object_cfg.name]
    q_w = obj.data.root_quat_w  # (N,4) wxyz in world

    up_local = torch.tensor([0.0, 0.0, 1.0], device=env.device, dtype=q_w.dtype).view(1, 3).repeat(q_w.shape[0], 1)
    up_w = quat_apply(q_w, up_local)  # (N,3)

    cos = torch.clamp(up_w[:, 2], -1.0, 1.0)  # dot with world +Z
    # tilt error in [0..2], 0 is perfect
    return 1.0 - cos

def object_upright_penalty_gated(
    env,
    object_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg,
    command_name: str,
    pos_std: float = 0.18,
    tilt_std: float = 0.25,
) -> torch.Tensor:
    """Only penalize tipping when the hand is near the goal (near the cylinder)."""
    # reuse your existing reach gate: 0 far, ~1 near
    gate = position_command_error_tanh(env, asset_cfg=asset_cfg, command_name=command_name, std=pos_std)
    tilt = object_upright_tilt_error(env, object_cfg)
    # smooth bounded penalty in [0,1]
    pen = torch.tanh(tilt / tilt_std)
    return gate * pen


def object_root_lin_vel_l2(env, object_cfg: SceneEntityCfg) -> torch.Tensor:
    obj = env.scene[object_cfg.name]
    v = obj.data.root_lin_vel_w  # (N,3)
    return torch.linalg.norm(v, dim=-1)

def object_root_ang_vel_l2(env, object_cfg: SceneEntityCfg) -> torch.Tensor:
    obj = env.scene[object_cfg.name]
    w = obj.data.root_ang_vel_w  # (N,3)
    return torch.linalg.norm(w, dim=-1)

def object_motion_penalty_gated(
    env,
    object_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg,
    command_name: str,
    pos_std: float = 0.18,
) -> torch.Tensor:
    gate = position_command_error_tanh(env, asset_cfg=asset_cfg, command_name=command_name, std=pos_std)
    lin = object_root_lin_vel_l2(env, object_cfg)
    ang = object_root_ang_vel_l2(env, object_cfg)
    # combine (you’ll weight it externally)
    return gate * (lin + 0.2 * ang)


def multi_contact_force_reward(
    env,
    sensor_names: list[str],
    min_force: float = 0.5,
    max_force: float = 20.0,
) -> torch.Tensor:
    """
    Returns [0..1] based on the MAX contact force across multiple sensors.
    Perfect as a single penalty that triggers if *any* finger/palm pushes.
    """
    device = env.device
    max_per_env = None

    for name in sensor_names:
        sensor = env.scene.sensors[name]
        forces = sensor.data.net_forces_w
        N = forces.shape[0]
        forces = forces.reshape(N, -1, 3)
        norms = torch.linalg.norm(forces, dim=-1)
        norms = torch.nan_to_num(norms, nan=0.0, posinf=0.0, neginf=0.0)
        f = norms.max(dim=1).values  # (N,)

        max_per_env = f if max_per_env is None else torch.maximum(max_per_env, f)

    denom = max(max_force - min_force, 1e-6)
    x = (max_per_env - min_force) / denom
    return torch.clamp(x, 0.0, 1.0)


def object_tipped(
    env,
    object_cfg: SceneEntityCfg,
    cos_thresh: float = 0.90,  # ~25.8 degrees
) -> torch.Tensor:
    """True if object tilt exceeds threshold."""
    obj = env.scene[object_cfg.name]
    q_w = obj.data.root_quat_w  # (N,4) wxyz

    up_local = torch.tensor([0.0, 0.0, 1.0], device=env.device, dtype=q_w.dtype).view(1, 3).repeat(q_w.shape[0], 1)
    up_w = quat_apply(q_w, up_local)
    cos = torch.clamp(up_w[:, 2], -1.0, 1.0)  # alignment with world +Z

    return cos < cos_thresh


def object_moved_too_far(
    env,
    object_cfg: SceneEntityCfg,
    dist_thresh: float = 0.02,  # 2 cm
) -> torch.Tensor:
    obj = env.scene[object_cfg.name]
    if not hasattr(env, "_cyl_spawn_pos_w"):
        # if we somehow didn't record spawn yet, don't kill the episode
        return torch.zeros((obj.data.root_pos_w.shape[0],), device=env.device, dtype=torch.bool)

    d = torch.linalg.norm(obj.data.root_pos_w - env._cyl_spawn_pos_w, dim=-1)
    return d > dist_thresh


def contact_force_band_reward(
    env,
    sensor_cfg: SceneEntityCfg,
    target_force: float = 4.0,   # N-ish (tune)
    tol: float = 2.0,            # softness (tune)
    min_contact: float = 0.4,    # ignore micro-grazes
    max_force: float = 20.0,
) -> torch.Tensor:
    f = _sensor_max_force_norm(env, sensor_cfg)
    f = torch.clamp(f, 0.0, max_force)

    in_contact = (f > min_contact).to(f.dtype)
    err = torch.abs(f - target_force)
    # reward in [0,1], peaks near target_force
    rew = 1.0 - torch.tanh(err / tol)
    return in_contact * rew



def ee_to_object_distance_tanh(
    env,
    ee_cfg: SceneEntityCfg,
    object_cfg: SceneEntityCfg,
    std: float = 0.08,
) -> torch.Tensor:
    # ee pos in robot root frame
    ee_pos_b = body_pos_in_root_frame(env, ee_cfg)

    # object pos in robot root frame
    robot = env.scene[ee_cfg.name]  # same asset ("robot")
    obj = env.scene[object_cfg.name]
    obj_pos_b, _ = subtract_frame_transforms(
        robot.data.root_pos_w, robot.data.root_quat_w,
        obj.data.root_pos_w, obj.data.root_quat_w
    )

    dist = torch.linalg.norm(ee_pos_b - obj_pos_b, dim=-1)
    return 1.0 - torch.tanh(dist / std)


# -----------------------------------------------------------------------------
# Hand-joint helper: keep fingers open until EE is at goal, then close (latched)
# -----------------------------------------------------------------------------

def reset_named_buffer(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    buffer_name: str = "_hand_closed",
    dtype: str = "bool",
) -> None:
    """Create/reset a per-env buffer on env resets (used for latching/counters).

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

    buf = getattr(env, buffer_name)
    # reset selected envs
    buf[env_ids] = 0 if buf.dtype != torch.bool else False


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
      1) **Joint ordering is guaranteed**: joint_ids are resolved **in the exact order of `hand_joint_cfg.joint_names`**.
         (Some helper APIs may return ids in a different/sorted order, which makes the hand do "weird shit".)
      2) Supports open/close targets as None | scalar | sequence | dict{name->val}.
      3) Clamps targets to the joints' position limits (if available).

    Units:
      - Isaac joint targets are **radians**.
      - If `close_pos_is_degrees=True`, close_pos values are interpreted as degrees and converted to radians.
    """
    robot: Articulation = env.scene[hand_joint_cfg.name]

    # -------------------------
    # Resolve joint ids *in the same order as joint_names*
    # -------------------------
    joint_names = getattr(hand_joint_cfg, "joint_names", None) or getattr(hand_joint_cfg, "joint_name", None)
    joint_ids = getattr(hand_joint_cfg, "joint_ids", None)

    # Build / cache name->id mapping once per env for speed
    if not hasattr(env, "_joint_name_to_id"):
        if not (hasattr(robot, "data") and hasattr(robot.data, "joint_names")):
            raise AttributeError("Cannot resolve joint ids: robot.data.joint_names not available.")
        names_all = list(robot.data.joint_names)
        env._joint_name_to_id = {n: i for i, n in enumerate(names_all)}

    name_to_id = env._joint_name_to_id

    if joint_names:
        # Force ids to match the provided order (this avoids 'sorted ids' bugs)
        try:
            ids_list = [int(name_to_id[n]) for n in joint_names]
        except KeyError as e:
            raise KeyError(f"Unknown joint name in hand_joint_cfg.joint_names: {e}") from e
        joint_ids = torch.tensor(ids_list, device=env.device, dtype=torch.int64)
    else:
        # No names given: fall back to existing joint_ids, but ensure it isn't a slice sentinel.
        if joint_ids is None or isinstance(joint_ids, slice):
            raise ValueError(
                "hand_joint_cfg must provide joint_names when joint_ids are not resolved."
            )
        # Ensure tensor
        if not isinstance(joint_ids, torch.Tensor):
            joint_ids = torch.tensor(list(joint_ids), device=env.device, dtype=torch.int64)

    num_joints = int(joint_ids.numel())
    dtype = robot.data.default_joint_pos.dtype

    # -------------------------
    # Convert open/close specs -> vectors (J,)
    # -------------------------
    def _spec_to_vec(spec, *, default_val: float = 0.0) -> torch.Tensor:
        if spec is None:
            return torch.full((num_joints,), float(default_val), device=env.device, dtype=dtype)

        if isinstance(spec, (float, int)):
            return torch.full((num_joints,), float(spec), device=env.device, dtype=dtype)

        if isinstance(spec, dict):
            if not joint_names:
                raise ValueError("Dict open_pos/close_pos requires hand_joint_cfg.joint_names to map keys.")
            v = torch.full((num_joints,), float(default_val), device=env.device, dtype=dtype)
            for j, n in enumerate(joint_names):
                if n in spec:
                    v[j] = float(spec[n])
            return v

        if isinstance(spec, (list, tuple)):
            if len(spec) != num_joints:
                raise ValueError(f"Target length mismatch: got {len(spec)} values, expected {num_joints}.")
            return torch.tensor(spec, device=env.device, dtype=dtype)

        raise TypeError(f"Unsupported target spec type: {type(spec)}")

    open_vec = _spec_to_vec(open_pos, default_val=0.0)
    close_vec = _spec_to_vec(close_pos, default_val=0.0)

    if close_pos_is_degrees:
        close_vec = torch.deg2rad(close_vec)

    # -------------------------
    # Trigger: reached goal OR stalled (depending on stop_only)
    # -------------------------
    if not hasattr(env, latch_buffer_name):
        setattr(env, latch_buffer_name, torch.zeros((env.num_envs,), device=env.device, dtype=torch.bool))
    hand_closed = getattr(env, latch_buffer_name)

    goal_pos_all = env.command_manager.get_command(command_name)[:, 0:3]
    ee_pos_all = body_pos_in_root_frame(env, ee_cfg)

    goal_pos = goal_pos_all[env_ids]
    ee_pos = ee_pos_all[env_ids]
    dist = torch.linalg.norm(ee_pos - goal_pos, dim=-1)
    reached_goal = dist < reach_threshold

    # step counter
    if not hasattr(env, step_buffer_name):
        setattr(env, step_buffer_name, torch.zeros((env.num_envs,), device=env.device, dtype=torch.int32))
    step_ctr = getattr(env, step_buffer_name)
    step_ctr[env_ids] = step_ctr[env_ids] + 1

    # stillness counter
    if not hasattr(env, stall_buffer_name):
        setattr(env, stall_buffer_name, torch.zeros((env.num_envs,), device=env.device, dtype=torch.int32))
    still_ctr = getattr(env, stall_buffer_name)

    # previous EE pos for dpos
    if not hasattr(env, prev_pos_buffer_name):
        buf = torch.zeros((env.num_envs, 3), device=env.device, dtype=ee_pos_all.dtype)
        buf[:] = ee_pos_all
        setattr(env, prev_pos_buffer_name, buf)
    prev_pos = getattr(env, prev_pos_buffer_name)

    prev = prev_pos[env_ids]
    dpos = torch.linalg.norm(ee_pos - prev, dim=-1)
    prev_pos[env_ids] = ee_pos

    # optional velocity-based stillness
    ee_body_id = _resolve_single_body_id(robot, ee_cfg)
    ee_speed = torch.linalg.norm(robot.data.body_lin_vel_w[:, ee_body_id], dim=-1)[env_ids]

    still_signal = (dpos < stall_pos_threshold) | (ee_speed < stall_vel_threshold)

    if stop_only:
        is_still = still_signal
    else:
        near_goal = dist < stall_near_threshold
        is_still = still_signal & near_goal

    # ignore first couple steps
    is_still = is_still & (step_ctr[env_ids] >= 3)

    still_ctr[env_ids] = torch.where(
        is_still,
        still_ctr[env_ids] + 1,
        torch.zeros_like(still_ctr[env_ids]),
    )

    stalled = (still_ctr[env_ids] >= stall_steps) & (step_ctr[env_ids] >= min_steps_before_stall)
    reached = stalled if stop_only else (reached_goal | stalled)

    hand_closed[env_ids] = hand_closed[env_ids] | reached
    closed_mask = hand_closed[env_ids]  # (B,)

    # -------------------------
    # Build per-env targets (B, J) and clamp to joint limits if present
    # -------------------------
    B = int(env_ids.numel())
    open_t = open_vec.view(1, -1).repeat(B, 1)
    close_t = close_vec.view(1, -1).repeat(B, 1)

    # Smoothly ramp from open -> close once closed_mask turns true.
    # This avoids large instantaneous penetration impulses that can make the object "pop"/teleport.
    if close_ramp_steps is None or close_ramp_steps <= 0:
        alpha = closed_mask.to(dtype).view(-1, 1)
    else:
        if not hasattr(env, ramp_buffer_name):
            setattr(env, ramp_buffer_name, torch.zeros((env.num_envs,), device=env.device, dtype=torch.int32))
        ramp_ctr = getattr(env, ramp_buffer_name)

        # If not yet closed: reset ramp to 0. If closed: increment up to close_ramp_steps.
        ramp_ctr[env_ids] = torch.where(
            closed_mask,
            torch.clamp(ramp_ctr[env_ids] + 1, max=int(close_ramp_steps)),
            torch.zeros_like(ramp_ctr[env_ids]),
        )

        alpha = (ramp_ctr[env_ids].to(dtype) / float(close_ramp_steps)).clamp(0.0, 1.0).view(-1, 1)

    targets = open_t + alpha * (close_t - open_t)


    # Clamp to limits (if available)
    if hasattr(robot.data, "joint_pos_limits") and robot.data.joint_pos_limits is not None:
        lim = robot.data.joint_pos_limits[env_ids][:, joint_ids, :]  # (B,J,2)
        lo = lim[..., 0]
        hi = lim[..., 1]
        targets = torch.clamp(targets, lo, hi)

    # Apply targets
    robot.set_joint_position_target(targets, joint_ids=joint_ids, env_ids=env_ids)


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


# ----------------------------------------------------------------------------
# Arm-lift helper: once the hand has closed (latched), smoothly move arm joints
# to a desired "lift" pose (e.g., to raise the object).
# ----------------------------------------------------------------------------

def lift_arm_to_joint_pose_when_hand_closed(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    asset_cfg: SceneEntityCfg,
    lift_joint_targets: dict[str, float],
    latch_buffer_name: str = "_hand_closed",
    targets_are_degrees: bool = True,
    # Optional: wait until the finger close-ramp finishes before starting the lift.
    wait_for_close_ramp: bool = True,
    close_ramp_buffer_name: str = "_hand_close_ramp",
    close_ramp_steps: int = 30,
    # Lift motion smoothing
    lift_ramp_steps: int = 60,
    ramp_buffer_name: str = "_arm_lift_ramp",
    start_buffer_name: str = "_arm_lift_start",
    active_buffer_name: str = "_arm_lift_active",
) -> None:
    """After the hand closes, move the specified arm joints to a lift pose.

    - Uses `latch_buffer_name` (set by your hand-closing event) as the main trigger.
    - If `wait_for_close_ramp=True` and the close-ramp counter exists, the lift starts *after* the fingers finish ramping.
    - Captures the starting arm joint pose at lift-start and ramps to the target pose.
    - Unspecified joints keep their starting value (so you can specify only the joints you care about).
    """
    robot: Articulation = env.scene[asset_cfg.name]

    joint_names = getattr(asset_cfg, "joint_names", None)
    if not joint_names or len(joint_names) == 0:
        raise ValueError("asset_cfg.joint_names must be provided for lift_arm_to_joint_pose_when_hand_closed.")

    # name -> id map (re-use if already built by other helpers)
    if not hasattr(env, "_joint_name_to_id"):
        if not (hasattr(robot, "data") and hasattr(robot.data, "joint_names")):
            raise AttributeError("Cannot resolve joint ids: robot.data.joint_names not available.")
        names_all = list(robot.data.joint_names)
        env._joint_name_to_id = {n: i for i, n in enumerate(names_all)}
    name_to_id = env._joint_name_to_id

    try:
        joint_ids = torch.tensor([int(name_to_id[n]) for n in joint_names], device=env.device, dtype=torch.int64)
    except KeyError as e:
        raise KeyError(f"Unknown joint name in asset_cfg.joint_names: {e}") from e

    J = int(joint_ids.numel())
    dtype = robot.data.default_joint_pos.dtype

    # latch buffer (created if missing)
    if not hasattr(env, latch_buffer_name):
        setattr(env, latch_buffer_name, torch.zeros((env.num_envs,), device=env.device, dtype=torch.bool))
    hand_closed_all = getattr(env, latch_buffer_name)
    latch_mask = hand_closed_all[env_ids]  # (B,)

    # Gate lift start until the close ramp finishes (if requested and available)
    can_lift_mask = latch_mask
    if wait_for_close_ramp and hasattr(env, close_ramp_buffer_name):
        close_ramp = getattr(env, close_ramp_buffer_name)[env_ids]
        can_lift_mask = can_lift_mask & (close_ramp >= int(close_ramp_steps))

    # buffers for lift ramping
    if not hasattr(env, active_buffer_name):
        setattr(env, active_buffer_name, torch.zeros((env.num_envs,), device=env.device, dtype=torch.bool))
    active_all = getattr(env, active_buffer_name)

    if not hasattr(env, ramp_buffer_name):
        setattr(env, ramp_buffer_name, torch.zeros((env.num_envs,), device=env.device, dtype=torch.int32))
    ramp_ctr_all = getattr(env, ramp_buffer_name)

    if not hasattr(env, start_buffer_name):
        setattr(env, start_buffer_name, torch.zeros((env.num_envs, J), device=env.device, dtype=dtype))
    start_all = getattr(env, start_buffer_name)

    # Reset lift state only when the latch is false (i.e., new episode or not yet triggered)
    if (~latch_mask).any():
        nc_ids = env_ids[(~latch_mask).nonzero(as_tuple=False).squeeze(-1)]
        active_all[nc_ids] = False
        ramp_ctr_all[nc_ids] = 0

    # Capture starting pose at the moment we are allowed to lift (after close ramp)
    newly_liftable = can_lift_mask & (~active_all[env_ids])
    if newly_liftable.any():
        new_ids = env_ids[newly_liftable.nonzero(as_tuple=False).squeeze(-1)]
        start_all[new_ids] = robot.data.joint_pos[new_ids][:, joint_ids]
        ramp_ctr_all[new_ids] = 0
        active_all[new_ids] = True

    # Apply lift only where allowed
    if not can_lift_mask.any():
        return

    lift_ids = env_ids[can_lift_mask.nonzero(as_tuple=False).squeeze(-1)]

    # compute ramp alpha
    if lift_ramp_steps is None or int(lift_ramp_steps) <= 0:
        alpha = torch.ones((lift_ids.numel(), 1), device=env.device, dtype=dtype)
    else:
        ramp_ctr_all[lift_ids] = torch.clamp(ramp_ctr_all[lift_ids] + 1, max=int(lift_ramp_steps))
        alpha = (ramp_ctr_all[lift_ids].to(dtype) / float(lift_ramp_steps)).clamp(0.0, 1.0).view(-1, 1)

    start_pos = start_all[lift_ids]  # (B,J)
    goal_pos = start_pos.clone()

    DEG2RAD = 0.017453292519943295
    for j, n in enumerate(joint_names):
        if n in lift_joint_targets:
            v = float(lift_joint_targets[n])
            if targets_are_degrees:
                v = v * DEG2RAD
            goal_pos[:, j] = v

    targets = start_pos + alpha * (goal_pos - start_pos)

    # clamp to joint limits (if available)
    if hasattr(robot.data, "joint_pos_limits") and robot.data.joint_pos_limits is not None:
        lim = robot.data.joint_pos_limits[lift_ids][:, joint_ids, :]  # (B,J,2)
        lo = lim[..., 0]
        hi = lim[..., 1]
        targets = torch.clamp(targets, lo, hi)

    robot.set_joint_position_target(targets, joint_ids=joint_ids, env_ids=lift_ids)


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

    # If latch doesn't exist yet, nothing to do
    if not hasattr(env, latch_buffer_name):
        return

    hand_closed_all = getattr(env, latch_buffer_name)
    mask = hand_closed_all[env_ids].clone()

    # Optionally wait for close ramp to finish (so goal changes AFTER fingers are actually closed)
    if wait_for_close_ramp and hasattr(env, close_ramp_buffer_name):
        ramp = getattr(env, close_ramp_buffer_name)[env_ids]
        mask = mask & (ramp >= int(close_ramp_steps))

    if not mask.any():
        return

    # Optional "only once" gating
    if once:
        if not hasattr(env, switched_buffer_name):
            setattr(env, switched_buffer_name, torch.zeros((env.num_envs,), device=env.device, dtype=torch.bool))
        switched = getattr(env, switched_buffer_name)

        eligible = mask & (~switched[env_ids])
        if not eligible.any():
            return

        use_env_ids = env_ids[eligible.nonzero(as_tuple=False).squeeze(-1)]
        switched[use_env_ids] = True
    else:
        use_env_ids = env_ids[mask.nonzero(as_tuple=False).squeeze(-1)]

    # Get command buffer and override xyz in-place
    cmd = env.command_manager.get_command(command_name)  # (num_envs, 7) typically
    tgt = torch.tensor(target_pos, device=cmd.device, dtype=cmd.dtype).view(1, 3).repeat(use_env_ids.numel(), 1)

    # If CommandManager exposes a setter, use it; otherwise in-place edit usually works.
    if hasattr(env.command_manager, "set_command"):
        cmd_new = cmd.clone()
        cmd_new[use_env_ids, 0:3] = tgt
        env.command_manager.set_command(command_name, cmd_new)
    else:
        cmd[use_env_ids, 0:3] = tgt


def ensure_goal_reachable_box_sphere(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    command_name: str,
    shoulder_cfg: SceneEntityCfg,
    x_range=(0.3, 0.4),
    y_range=(-0.05, 0.3),
    z_range=(0.05, 0.3),
    r_min: float = 0.12,
    r_max: float = 0.48,
    max_tries: int = 80,
) -> None:
    """Keep the commanded goal inside:
      1) the AABB box ranges (x/y/z)
      2) a spherical shell [r_min, r_max] centered at the shoulder (in robot root frame)

    If invalid, resample within the box until valid (rejection sampling).
    """
    if env_ids.numel() == 0:
        return

    # command buffer: expected shape (num_envs, >=3)
    cmd = env.command_manager.get_command(command_name)
    if cmd.shape[-1] < 3:
        raise ValueError(f"Command '{command_name}' must have at least xyz (got shape {cmd.shape}).")

    # Robot articulation (shoulder_cfg.name should be "robot")
    robot: Articulation = env.scene[shoulder_cfg.name]

    # Shoulder position in robot root frame (per env)
    shoulder_id = _resolve_single_body_id(robot, shoulder_cfg)

    root_pos_w = robot.data.root_pos_w[env_ids]
    root_quat_w = robot.data.root_quat_w[env_ids]

    shoulder_pos_w = robot.data.body_pos_w[env_ids, shoulder_id]
    shoulder_quat_w = robot.data.body_quat_w[env_ids, shoulder_id]

    shoulder_pos_b, _ = subtract_frame_transforms(root_pos_w, root_quat_w, shoulder_pos_w, shoulder_quat_w)

    # Current goal xyz in root frame for these envs
    goal_pos = cmd[env_ids, 0:3]

    x0, x1 = float(x_range[0]), float(x_range[1])
    y0, y1 = float(y_range[0]), float(y_range[1])
    z0, z1 = float(z_range[0]), float(z_range[1])

    # validity checks
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

    # envs that need fixing
    bad_local = (~ok).nonzero(as_tuple=False).squeeze(-1)  # indices into env_ids
    bad_env_ids = env_ids[bad_local]
    B = int(bad_env_ids.numel())

    # shoulder positions for just the bad envs
    shoulder_bad = shoulder_pos_b[bad_local]

    device = cmd.device
    dtype = cmd.dtype

    new_pos = torch.empty((B, 3), device=device, dtype=dtype)
    accepted = torch.zeros((B,), device=device, dtype=torch.bool)

    # rejection sampling loop
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

    # fallback if something went wrong: project random directions onto mid-shell and clamp into box
    if not accepted.all():
        m = (~accepted).nonzero(as_tuple=False).squeeze(-1)
        n = int(m.numel())
        dirs = torch.randn((n, 3), device=device, dtype=dtype)
        dirs = dirs / (torch.linalg.norm(dirs, dim=-1, keepdim=True) + 1e-8)

        r_mid = 0.5 * (float(r_min) + float(r_max))
        p = shoulder_bad[m] + r_mid * dirs

        # clamp into your AABB
        p[:, 0] = torch.clamp(p[:, 0], x0, x1)
        p[:, 1] = torch.clamp(p[:, 1], y0, y1)
        p[:, 2] = torch.clamp(p[:, 2], z0, z1)

        new_pos[m] = p

    # Write back into the command buffer
    if hasattr(env.command_manager, "set_command"):
        cmd_new = cmd.clone()
        cmd_new[bad_env_ids, 0:3] = new_pos
        env.command_manager.set_command(command_name, cmd_new)
    else:
        cmd[bad_env_ids, 0:3] = new_pos


def success_bonus(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    command_name: str,
    threshold: float = 0.05,
) -> torch.Tensor:
    """Return 1.0 when within threshold of goal, else 0.0 (in base/root frame)."""
    done = reached_ee_position(env, asset_cfg=asset_cfg, command_name=command_name, threshold=threshold)
    # Match dtype/device nicely
    ee_pos = body_pos_in_root_frame(env, asset_cfg)
    return done.to(dtype=ee_pos.dtype)

def position_command_error_tanh(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    command_name: str,
    std: float = 0.15,
    eps: float = 1e-6,
    squared: bool = False,
) -> torch.Tensor:
    """Stable bounded tracking reward in [0, 1].

    r = 1 - tanh( (dist/std) )      (default)
    Optionally: r = 1 - tanh( (dist/std)^2 )  (sharper near goal)
    """
    dist = position_command_error(env, asset_cfg=asset_cfg, command_name=command_name)

    # std as tensor on correct device/dtype + clamp to avoid divide-by-zero
    std_t = torch.as_tensor(std, device=dist.device, dtype=dist.dtype).clamp_min(eps)

    x = dist / std_t
    if squared:
        x = x * x

    rew = 1.0 - torch.tanh(x)

    # Hard safety: kill NaNs/Infs
    rew = torch.nan_to_num(rew, nan=0.0, posinf=0.0, neginf=0.0)

    # Keep in [0,1] (tanh guarantees this, but clamp is cheap insurance)
    return torch.clamp(rew, 0.0, 1.0)


def contact_force_reward_any(
    env,
    sensor_cfgs: list[SceneEntityCfg],
    min_force: float = 1.0,
    max_force: float = 50.0,
) -> torch.Tensor:
    """Like contact_force_reward, but takes the max contact across multiple sensors."""
    if len(sensor_cfgs) == 0:
        return torch.zeros(env.num_envs, device=env.device)

    f_list = [_sensor_max_force_norm(env, s_cfg) for s_cfg in sensor_cfgs]  # (N,) each
    f = torch.stack(f_list, dim=0).max(dim=0).values  # (N,)

    denom = max(max_force - min_force, 1e-6)
    x = (f - min_force) / denom
    return torch.clamp(x, 0.0, 1.0)


def contact_force_exceeds_any(env, sensor_cfgs: list[SceneEntityCfg], force_thresh: float):
    if len(sensor_cfgs) == 0:
        return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

    # max norm across sensors
    f_list = [_sensor_max_force_norm(env, s_cfg) for s_cfg in sensor_cfgs]
    f = torch.stack(f_list, dim=0).max(dim=0).values
    return f > force_thresh

def ee_lin_speed_w(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """End-effector linear speed in world frame (m/s)."""
    asset: Articulation = env.scene[asset_cfg.name]
    body_id = _resolve_single_body_id(asset, asset_cfg)
    v = asset.data.body_lin_vel_w[:, body_id]  # (N,3)
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
    """True when EE is within pos_threshold AND speed < speed_threshold for dwell_steps consecutive steps."""
    # condition
    goal_pos = env.command_manager.get_command(command_name)[:, 0:3]
    ee_pos = body_pos_in_root_frame(env, asset_cfg)
    dist = torch.linalg.norm(ee_pos - goal_pos, dim=-1)

    speed = ee_lin_speed_w(env, asset_cfg)
    ok = (dist < pos_threshold) & (speed < speed_threshold)

    # per-env dwell counter
    if not hasattr(env, buffer_name):
        setattr(env, buffer_name, torch.zeros((env.num_envs,), device=env.device, dtype=torch.int32))
    ctr = getattr(env, buffer_name)

    ctr[:] = torch.where(ok, torch.clamp(ctr + 1, max=int(dwell_steps)), torch.zeros_like(ctr))
    return ctr >= int(dwell_steps)


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

    # Need the goal-switched latch
    if not hasattr(env, goal_switched_buffer_name):
        return
    goal_switched_all = getattr(env, goal_switched_buffer_name)

    # Only consider envs where goal has already switched
    switched_mask = goal_switched_all[env_ids].clone()
    if not switched_mask.any():
        return

    # Distance-to-current-command goal (in robot/root frame)
    goal_pos_all = env.command_manager.get_command(command_name)[:, 0:3]
    ee_pos_all = body_pos_in_root_frame(env, ee_cfg)

    dist = torch.linalg.norm(ee_pos_all[env_ids] - goal_pos_all[env_ids], dim=-1)
    reached_mask = dist < reach_threshold

    can_open = switched_mask & reached_mask
    if not can_open.any() and not hasattr(env, opened_buffer_name):
        return

    robot: Articulation = env.scene[hand_joint_cfg.name]

    # Resolve joint ids in the SAME order as joint_names (same trick as your close helper)
    joint_names = getattr(hand_joint_cfg, "joint_names", None)
    if not joint_names or len(joint_names) == 0:
        raise ValueError("hand_joint_cfg.joint_names must be provided for open_hand_once_when_at_current_goal.")

    if not hasattr(env, "_joint_name_to_id"):
        names_all = list(robot.data.joint_names)
        env._joint_name_to_id = {n: i for i, n in enumerate(names_all)}
    name_to_id = env._joint_name_to_id

    joint_ids = torch.tensor([int(name_to_id[n]) for n in joint_names], device=env.device, dtype=torch.int64)
    J = int(joint_ids.numel())
    dtype = robot.data.default_joint_pos.dtype

    # Build open target vector (J,)
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

    # Latch buffers
    if not hasattr(env, opened_buffer_name):
        setattr(env, opened_buffer_name, torch.zeros((env.num_envs,), device=env.device, dtype=torch.bool))
    opened_all = getattr(env, opened_buffer_name)

    if not hasattr(env, ramp_buffer_name):
        setattr(env, ramp_buffer_name, torch.zeros((env.num_envs,), device=env.device, dtype=torch.int32))
    ramp_ctr_all = getattr(env, ramp_buffer_name)

    if not hasattr(env, start_buffer_name):
        setattr(env, start_buffer_name, torch.zeros((env.num_envs, J), device=env.device, dtype=dtype))
    start_all = getattr(env, start_buffer_name)

    # Trigger newly-opened envs
    newly = can_open & (~opened_all[env_ids])
    if newly.any():
        new_ids = env_ids[newly.nonzero(as_tuple=False).squeeze(-1)]
        start_all[new_ids] = robot.data.joint_pos[new_ids][:, joint_ids]
        ramp_ctr_all[new_ids] = 0
        opened_all[new_ids] = True

    # Apply open targets for all envs that are opened (keeps hand open)
    opened_mask = opened_all[env_ids]
    if not opened_mask.any():
        return
    use_ids = env_ids[opened_mask.nonzero(as_tuple=False).squeeze(-1)]

    # Ramp from captured start -> open_vec
    if open_ramp_steps is None or int(open_ramp_steps) <= 0:
        alpha = torch.ones((use_ids.numel(), 1), device=env.device, dtype=dtype)
    else:
        ramp_ctr_all[use_ids] = torch.clamp(ramp_ctr_all[use_ids] + 1, max=int(open_ramp_steps))
        alpha = (ramp_ctr_all[use_ids].to(dtype) / float(open_ramp_steps)).clamp(0.0, 1.0).view(-1, 1)

    start_pos = start_all[use_ids]  # (B,J)
    goal_pos = open_vec.view(1, -1).repeat(use_ids.numel(), 1)
    targets = start_pos + alpha * (goal_pos - start_pos)

    # Clamp to joint limits if available
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

    # Create or fix shape
    need_new = True
    if hasattr(env, buffer_name):
        buf = getattr(env, buffer_name)
        if isinstance(buf, torch.Tensor) and buf.ndim == 2 and buf.shape[0] == env.num_envs and buf.shape[1] == J:
            need_new = False

    if need_new:
        setattr(env, buffer_name, torch.zeros((env.num_envs, J), device=env.device, dtype=dtype))

    # Reset only the envs that are resetting
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
    # Detect "just reset" envs (support both common buffer names)
    if hasattr(env, "reset_buf"):
        just_reset = env.reset_buf[env_ids].to(torch.bool)
    elif hasattr(env, "episode_length_buf"):
        just_reset = (env.episode_length_buf[env_ids] == 0)
    else:
        return

    if not just_reset.any():
        return

    ids = env_ids[just_reset.nonzero(as_tuple=False).squeeze(-1)]

    # Re-arm the whole open-after-goal2 pipeline
    reset_named_buffer(env, ids, buffer_name=goal_switched_buffer_name, dtype="bool")
    reset_named_buffer(env, ids, buffer_name=opened_buffer_name, dtype="bool")
    reset_named_buffer(env, ids, buffer_name=ramp_buffer_name, dtype="int")
    reset_hand_open_start_buffer(env, ids, hand_joint_cfg=hand_joint_cfg, buffer_name=start_buffer_name)
