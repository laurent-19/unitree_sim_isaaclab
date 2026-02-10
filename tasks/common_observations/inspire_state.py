# Copyright (c) 2025, Unitree Robotics Co., Ltd. All Rights Reserved.
# License: Apache License, Version 2.0  
"""
Inspire Hand state observation with contact force sensing
Supports FORCE_ACT (register 1582) data from Isaac Lab simulation
"""      

from __future__ import annotations

import torch
from typing import TYPE_CHECKING, Optional
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
    "contact_force_buf": None,  # Buffer for contact forces (FORCE_ACT)
    "dds_last_ms": 0,
    "dds_min_interval_ms": 20,
}

# Finger body indices for contact force sensing (maps to FORCE_ACT register 1582)
# These are the distal/tip links of each finger that contact objects
INSPIRE_FINGER_BODIES = {
    # Left hand finger tips (6 sensors: index, middle, ring, pinky, thumb, palm)
    "left": [
        "L_index_intermediate",      # Index finger tip
        "L_middle_intermediate",     # Middle finger tip
        "L_ring_intermediate",       # Ring finger tip
        "L_pinky_intermediate",      # Pinky finger tip
        "L_thumb_distal",            # Thumb tip
        "left_wrist_roll_link",      # Palm (approximate)
    ],
    # Right hand finger tips
    "right": [
        "R_index_intermediate",
        "R_middle_intermediate", 
        "R_ring_intermediate",
        "R_pinky_intermediate",
        "R_thumb_distal",
        "right_wrist_roll_link",
    ]
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

# global variable to cache the DDS instance
_inspire_dds = None
_dds_initialized = False

def _get_inspire_dds_instance():
    """get the DDS instance, delay initialization"""
    global _inspire_dds, _dds_initialized
    
    if not _dds_initialized or _inspire_dds is None:
        try:
            # dynamically import the DDS module
            sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), 'dds'))
            from dds.dds_master import dds_manager
            _inspire_dds = dds_manager.get_object("inspire")
            print("[Observations] DDS communication instance obtained")
            
            # register the cleanup function
            import atexit
            def cleanup_dds():
                try:
                    if _inspire_dds:
                        dds_manager.unregister_object("inspire")
                        print("[gripper_state] DDS communication closed correctly")
                except Exception as e:
                    print(f"[gripper_state] Error closing DDS: {e}")
            atexit.register(cleanup_dds)
            
        except Exception as e:
            print(f"[Observations] Failed to get DDS instances: {e}")
            _inspire_dds = None
        
        _dds_initialized = True
    
    return _inspire_dds



def get_robot_inspire_joint_states(
    env: ManagerBasedRLEnv,
    enable_dds: bool = True,
) -> torch.Tensor:
    """get the robot gripper joint states and publish them to DDS
    
    Args:
        env: ManagerBasedRLEnv - reinforcement learning environment instance
        enable_dds: bool - whether to enable the DDS publish function
    
    Returns:
        torch.Tensor: joint positions for the inspire hand
    """
    # get the gripper joint states
    joint_pos = env.scene["robot"].data.joint_pos
    joint_vel = env.scene["robot"].data.joint_vel  
    joint_torque = env.scene["robot"].data.applied_torque
    device = joint_pos.device
    batch = joint_pos.shape[0]
    

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
        _obs_cache["contact_force_buf"] = torch.zeros(batch, 12, device=device, dtype=joint_pos.dtype)  # 6 per hand
        _obs_cache["batch"] = batch

    idx_batch = _obs_cache["inspire_idx_batch"]
    pos_buf = _obs_cache["pos_buf"]
    vel_buf = _obs_cache["vel_buf"]
    torque_buf = _obs_cache["torque_buf"]
    contact_force_buf = _obs_cache["contact_force_buf"]


    try:
        torch.gather(joint_pos, 1, idx_batch, out=pos_buf)
        torch.gather(joint_vel, 1, idx_batch, out=vel_buf)
        torch.gather(joint_torque, 1, idx_batch, out=torque_buf)
    except TypeError:
        pos_buf.copy_(torch.gather(joint_pos, 1, idx_batch))
        vel_buf.copy_(torch.gather(joint_vel, 1, idx_batch))
        torque_buf.copy_(torch.gather(joint_torque, 1, idx_batch))
    
    # Try to get contact force data (FORCE_ACT - register 1582)
    contact_forces = _get_finger_contact_forces(env, device, batch)
    if contact_forces is not None:
        contact_force_buf.copy_(contact_forces)
    
    # publish to DDS (only publish the data of the first environment)
    if enable_dds and len(pos_buf) > 0:
        try:
            import time
            now_ms = int(time.time() * 1000)
            if now_ms - _obs_cache["dds_last_ms"] >= _obs_cache["dds_min_interval_ms"]:
                inspire_dds = _get_inspire_dds_instance()
                if inspire_dds:
                    pos = pos_buf[0].contiguous().cpu().numpy()
                    vel = vel_buf[0].contiguous().cpu().numpy()
                    torque = torque_buf[0].contiguous().cpu().numpy()
                    forces = contact_force_buf[0].contiguous().cpu().numpy()
                    # write the gripper state to shared memory (including contact forces)
                    inspire_dds.write_inspire_state(pos, vel, torque, forces)
                    _obs_cache["dds_last_ms"] = now_ms
        except Exception as e:
            print(f"[gripper_state] Failed to write to shared memory: {e}")
    
    return pos_buf


def _get_finger_contact_forces(
    env: ManagerBasedRLEnv,
    device: torch.device,
    batch: int,
) -> Optional[torch.Tensor]:
    """Extract contact forces for each finger from the contact sensor.
    
    This maps to the FORCE_ACT register (1582) on the real Inspire hand.
    Returns force magnitudes for: index, middle, ring, pinky, thumb, palm (per hand)
    
    Args:
        env: The environment instance
        device: Torch device
        batch: Batch size
        
    Returns:
        torch.Tensor: Contact force magnitudes [batch, 12] (6 per hand)
                      or None if contact sensor not available
    """
    try:
        # Check if contact sensor exists in scene
        if not hasattr(env.scene, "contact_forces"):
            return None
            
        contact_sensor = env.scene["contact_forces"]
        if contact_sensor is None:
            return None
        
        # Get net contact forces - shape: [batch, num_bodies, 3]
        net_forces = contact_sensor.data.net_forces_w
        if net_forces is None:
            return None
        
        # Compute force magnitudes
        force_magnitudes = torch.norm(net_forces, dim=-1)  # [batch, num_bodies]
        
        # Try to find finger body indices
        # Get body names from the robot
        robot = env.scene["robot"]
        body_names = robot.data.body_names if hasattr(robot.data, 'body_names') else []
        
        # Initialize output tensor [batch, 12] - 6 forces per hand
        finger_forces = torch.zeros(batch, 12, device=device, dtype=force_magnitudes.dtype)
        
        # Map body names to force indices
        for hand_idx, hand_side in enumerate(["left", "right"]):
            finger_bodies = INSPIRE_FINGER_BODIES.get(hand_side, [])
            for finger_idx, body_name in enumerate(finger_bodies):
                # Find body index by partial name match
                for body_idx, name in enumerate(body_names):
                    if body_name.lower() in name.lower():
                        if body_idx < force_magnitudes.shape[1]:
                            finger_forces[:, hand_idx * 6 + finger_idx] = force_magnitudes[:, body_idx]
                        break
        
        return finger_forces
        
    except Exception as e:
        # Silently fail - contact sensor may not be configured
        return None


def get_inspire_contact_forces(
    env: ManagerBasedRLEnv,
    enable_dds: bool = False,
) -> torch.Tensor:
    """Get contact force observations for the Inspire hand fingers.
    
    This is a separate observation term that can be used independently.
    Maps to FORCE_ACT register (1582) on real hardware.
    
    Args:
        env: ManagerBasedRLEnv - reinforcement learning environment instance
        enable_dds: bool - whether to enable the DDS publish (usually False, 
                          as forces are published with joint states)
    
    Returns:
        torch.Tensor: Contact force magnitudes [batch, 12]
                      Layout: [L_index, L_middle, L_ring, L_pinky, L_thumb, L_palm,
                               R_index, R_middle, R_ring, R_pinky, R_thumb, R_palm]
    """
    device = env.scene["robot"].data.joint_pos.device
    batch = env.scene["robot"].data.joint_pos.shape[0]
    
    forces = _get_finger_contact_forces(env, device, batch)
    if forces is None:
        return torch.zeros(batch, 12, device=device)
    
    return forces


