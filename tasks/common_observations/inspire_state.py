# Copyright (c) 2025, Unitree Robotics Co., Ltd. All Rights Reserved.
# License: Apache License, Version 2.0  
"""
gripper state
"""      

from __future__ import annotations

import torch
from typing import TYPE_CHECKING
import sys
import os
if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


import torch


_obs_cache = {
    "device": None,
    "batch": None,
    "inspire_idx_t": None,
    "inspire_idx_batch": None,
    "pos_buf": None,
    "vel_buf": None,
    "torque_buf": None,
    "dds_last_ms": 0,
    "dds_min_interval_ms": 20,
}

def get_robot_girl_joint_names() -> list[str]:
    return [
        "R_pinky_proximal_joint",
        "R_ring_proximal_joint",
        "R_middle_proximal_joint",
        "R_index_proximal_joint",
        "R_thumb_proximal_pitch_joint",
        "R_thumb_proximal_yaw_joint",
        "L_pinky_proximal_joint",
        "L_ring_proximal_joint",
        "L_middle_proximal_joint",
        "L_index_proximal_joint",
        "L_thumb_proximal_pitch_joint",
        "L_thumb_proximal_yaw_joint",
    ]

# global variable to cache the DDS instances for left and right hands
_inspire_dds = {'l': None, 'r': None}
_dds_initialized = False

def _get_inspire_dds_instances():
    """get the DDS instances for both hands, delay initialization"""
    global _inspire_dds, _dds_initialized

    if not _dds_initialized:
        try:
            # dynamically import the DDS module
            sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), 'dds'))
            from dds.dds_master import dds_manager

            _inspire_dds['l'] = dds_manager.get_object("inspire_l")
            _inspire_dds['r'] = dds_manager.get_object("inspire_r")

            if _inspire_dds['l'] or _inspire_dds['r']:
                print(f"[inspire_state] DDS instances obtained: L={_inspire_dds['l'] is not None}, R={_inspire_dds['r'] is not None}")

            # register the cleanup function
            import atexit
            def cleanup_dds():
                try:
                    from dds.dds_master import dds_manager
                    for side in ['l', 'r']:
                        if _inspire_dds[side]:
                            dds_manager.unregister_object(f"inspire_{side}")
                    print("[inspire_state] DDS communication closed correctly")
                except Exception as e:
                    print(f"[inspire_state] Error closing DDS: {e}")
            atexit.register(cleanup_dds)

            # Only mark as initialized if we got at least one instance
            if _inspire_dds['l'] or _inspire_dds['r']:
                _dds_initialized = True

        except Exception as e:
            print(f"[inspire_state] Failed to get DDS instances: {e}")

    return _inspire_dds



def _get_contact_forces(env, side: str) -> torch.Tensor:
    """Get contact forces from fingertip sensors for force_act.

    Args:
        env: ManagerBasedRLEnv instance
        side: 'left' or 'right'

    Returns:
        Tensor of shape (batch, 6) with force magnitudes in Newtons for each finger
        Order: pinky, ring, middle, index, thumb_bend, thumb_rot (thumb uses same force for both)
    """
    fingertip_sensor = f"{side}_fingertip_contacts"  # index, middle, ring, pinky intermediate links
    thumb_sensor = f"{side}_thumb_contacts"  # thumb distal link
    batch = env.num_envs
    device = env.device

    # Default to zeros
    forces = torch.zeros(batch, 6, device=device)

    # Get finger forces (pinky, ring, middle, index from intermediate links)
    # Sensor body order depends on how regex matches - typically alphabetical
    # Expected order: index, middle, pinky, ring (alphabetical)
    if fingertip_sensor in env.scene.sensors:
        net_forces = env.scene[fingertip_sensor].data.net_forces_w
        # Map sensor indices to our order: pinky(0), ring(1), middle(2), index(3)
        # Alphabetical regex match: index(0), middle(1), pinky(2), ring(3)
        sensor_to_force_idx = {0: 3, 1: 2, 2: 0, 3: 1}  # sensor_idx -> force_idx
        for sensor_idx, force_idx in sensor_to_force_idx.items():
            if sensor_idx < net_forces.shape[1]:
                force_magnitude = torch.norm(net_forces[:, sensor_idx, :], dim=-1)
                forces[:, force_idx] = force_magnitude

    # Get thumb force (from distal link)
    if thumb_sensor in env.scene.sensors:
        thumb_forces = env.scene[thumb_sensor].data.net_forces_w
        if thumb_forces.shape[1] > 0:
            thumb_force = torch.norm(thumb_forces[:, 0, :], dim=-1)
            forces[:, 4] = thumb_force  # thumb_bend
            forces[:, 5] = thumb_force  # thumb_rot (same force)

    return forces


def get_robot_inspire_joint_states(
    env: ManagerBasedRLEnv,
    enable_dds: bool = True,
) -> torch.Tensor:
    """get the robot gripper joint states and publish them to DDS

    Args:
        env: ManagerBasedRLEnv - reinforcement learning environment instance
        enable_dds: bool - whether to enable the DDS publish function

    返回:
        torch.Tensor
    """
    # get the gripper joint states
    joint_pos = env.scene["robot"].data.joint_pos
    joint_vel = env.scene["robot"].data.joint_vel
    joint_torque = env.scene["robot"].data.applied_torque
    device = joint_pos.device
    batch = joint_pos.shape[0]

    # Get contact forces from fingertip sensors (for force_act)
    right_contact_forces = _get_contact_forces(env, "right")
    left_contact_forces = _get_contact_forces(env, "left")
    

    global _obs_cache
    if _obs_cache["device"] != device or _obs_cache["inspire_idx_t"] is None:
        inspire_joint_indices = [36, 37, 35, 34, 48, 38, 31, 32, 30, 29, 43, 33]
        _obs_cache["inspire_idx_t"] = torch.tensor(inspire_joint_indices, dtype=torch.long, device=device)
        _obs_cache["device"] = device
        _obs_cache["batch"] = None
    idx_t = _obs_cache["inspire_idx_t"]
    n = idx_t.numel()


    if _obs_cache["batch"] != batch or _obs_cache["inspire_idx_batch"] is None:
        _obs_cache["inspire_idx_batch"] = idx_t.unsqueeze(0).expand(batch, n)
        _obs_cache["pos_buf"] = torch.empty(batch, n, device=device, dtype=joint_pos.dtype)
        _obs_cache["vel_buf"] = torch.empty(batch, n, device=device, dtype=joint_pos.dtype)
        _obs_cache["torque_buf"] = torch.empty(batch, n, device=device, dtype=joint_pos.dtype)
        _obs_cache["batch"] = batch

    idx_batch = _obs_cache["inspire_idx_batch"]
    pos_buf = _obs_cache["pos_buf"]
    vel_buf = _obs_cache["vel_buf"]
    torque_buf = _obs_cache["torque_buf"]


    try:
        torch.gather(joint_pos, 1, idx_batch, out=pos_buf)
        torch.gather(joint_vel, 1, idx_batch, out=vel_buf)
        torch.gather(joint_torque, 1, idx_batch, out=torque_buf)
    except TypeError:
        pos_buf.copy_(torch.gather(joint_pos, 1, idx_batch))
        vel_buf.copy_(torch.gather(joint_vel, 1, idx_batch))
        torque_buf.copy_(torch.gather(joint_torque, 1, idx_batch))
    
    # publish to DDS (only publish the data of the first environment)
    if enable_dds and len(pos_buf) > 0:
        try:
            import time
            now_ms = int(time.time() * 1000)
            if now_ms - _obs_cache["dds_last_ms"] >= _obs_cache["dds_min_interval_ms"]:
                inspire_dds_instances = _get_inspire_dds_instances()

                pos = pos_buf[0].contiguous().cpu().numpy()
                vel = vel_buf[0].contiguous().cpu().numpy()
                torque = torque_buf[0].contiguous().cpu().numpy()

                # Get contact forces (in Newtons) for force_act
                r_forces = right_contact_forces[0].contiguous().cpu().numpy()
                l_forces = left_contact_forces[0].contiguous().cpu().numpy()

                # Right hand: indices 0-5, Left hand: indices 6-11
                if inspire_dds_instances['r']:
                    inspire_dds_instances['r'].write_inspire_state(
                        pos[:6], vel[:6], torque[:6], r_forces
                    )
                if inspire_dds_instances['l']:
                    inspire_dds_instances['l'].write_inspire_state(
                        pos[6:12], vel[6:12], torque[6:12], l_forces
                    )

                _obs_cache["dds_last_ms"] = now_ms
        except Exception as e:
            print(f"[inspire_state] Failed to write to shared memory: {e}")
    
    return pos_buf


