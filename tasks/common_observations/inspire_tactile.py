# Copyright (c) 2025, Unitree Robotics Co., Ltd. All Rights Reserved.
# License: Apache License, Version 2.0
"""
Inspire Hand tactile state extraction and DDS publishing.

Extracts contact forces from ContactSensor on finger links and converts
them to taxel arrays matching the real inspire_hand_touch DDS format.
"""

from __future__ import annotations

import torch
import numpy as np
import sys
import os
from typing import TYPE_CHECKING, Optional, Dict, Any

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

from .tactile_mapping import (
    force_to_taxel_grid,
    flatten_taxel_grid,
    TACTILE_GRIDS,
)

# Finger name mapping: simulation link prefix -> real IDL field prefix
FINGER_MAP = {
    "pinky": "fingerone",
    "ring": "fingertwo",
    "middle": "fingerthree",
    "index": "fingerfour",
    "thumb": "fingerfive",
}

# Ordered list matching sensor body order
FINGER_ORDER = ["pinky", "ring", "middle", "index", "thumb"]

# Module-level cache for DDS and timing
_tactile_cache = {
    "dds": None,
    "dds_initialized": False,
    "last_publish_ms": 0,
    "publish_interval_ms": 20,  # 50Hz publishing rate
}


def _get_touch_dds():
    """Get the inspire_touch DDS instance with lazy initialization."""
    global _tactile_cache

    # Always try to get DDS if we don't have it yet
    if _tactile_cache["dds"] is None:
        try:
            sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), 'dds'))
            from dds.dds_master import dds_manager
            _tactile_cache["dds"] = dds_manager.get_object("inspire_touch")
            if _tactile_cache["dds"]:
                print("[inspire_tactile] DDS touch instance obtained")
            # Only log failure once
            elif not _tactile_cache["dds_initialized"]:
                print("[inspire_tactile] DDS touch instance not found (will retry)")
        except Exception as e:
            if not _tactile_cache["dds_initialized"]:
                print(f"[inspire_tactile] DDS init failed: {e}")
        _tactile_cache["dds_initialized"] = True

    return _tactile_cache["dds"]


def _check_sensor_available(env: ManagerBasedRLEnv, sensor_name: str) -> bool:
    """Check if a sensor is available in the scene."""
    return sensor_name in env.scene.sensors


def get_inspire_tactile_state(
    env: ManagerBasedRLEnv,
    enable_dds: bool = True,
) -> torch.Tensor:
    """Extract tactile data from contact sensors and optionally publish to DDS.

    This function reads contact forces from ContactSensor instances attached
    to the Inspire hand finger links. Forces are converted to taxel arrays
    for DDS publishing (first environment only) and returned as flattened
    force vectors for RL observations.

    Args:
        env: Isaac Lab ManagerBasedRLEnv instance
        enable_dds: Whether to publish tactile data via DDS

    Returns:
        torch.Tensor: Flattened contact forces for all finger sensors
                     Shape: (num_envs, total_force_components)
    """
    device = env.device
    batch = env.num_envs

    # Collect force tensors from available sensors
    force_tensors = []
    sensor_data = {}

    # Try to get left hand sensors
    for sensor_name, prefix in [
        ("left_fingertip_contacts", "left_tips"),
        ("left_finger_pad_contacts", "left_pads"),
        ("left_palm_contacts", "left_palm"),
    ]:
        if _check_sensor_available(env, sensor_name):
            forces = env.scene[sensor_name].data.net_forces_w
            force_tensors.append(forces.reshape(batch, -1))
            sensor_data[prefix] = forces

    # Try to get right hand sensors
    for sensor_name, prefix in [
        ("right_fingertip_contacts", "right_tips"),
        ("right_finger_pad_contacts", "right_pads"),
        ("right_palm_contacts", "right_palm"),
    ]:
        if _check_sensor_available(env, sensor_name):
            forces = env.scene[sensor_name].data.net_forces_w
            force_tensors.append(forces.reshape(batch, -1))
            sensor_data[prefix] = forces

    # If no sensors found, return empty tensor
    if not force_tensors:
        return torch.zeros(batch, 1, device=device)

    # Publish to DDS for first environment
    if enable_dds and batch > 0 and "left_tips" in sensor_data:
        _publish_tactile_to_dds(sensor_data)
    elif not _tactile_cache.get("no_data_logged"):
        # Log once why publishing isn't happening
        print(f"[inspire_tactile] Not publishing: enable_dds={enable_dds}, batch={batch}, left_tips={'left_tips' in sensor_data}, sensors={list(sensor_data.keys())}")
        _tactile_cache["no_data_logged"] = True

    # Concatenate all forces for RL observation
    all_forces = torch.cat(force_tensors, dim=-1)
    return all_forces


def _publish_tactile_to_dds(sensor_data: Dict[str, torch.Tensor]):
    """Publish tactile data to DDS with rate limiting.

    Args:
        sensor_data: Dictionary mapping sensor names to force tensors
    """
    import time
    global _tactile_cache

    now_ms = int(time.time() * 1000)
    if now_ms - _tactile_cache["last_publish_ms"] < _tactile_cache["publish_interval_ms"]:
        return

    dds = _get_touch_dds()
    if not dds:
        if not _tactile_cache.get("no_dds_logged"):
            print("[inspire_tactile] DDS object not available for publishing")
            _tactile_cache["no_dds_logged"] = True
        return

    try:
        tactile_data = _build_tactile_message(sensor_data)
        dds.write_tactile_data(tactile_data)
        _tactile_cache["last_publish_ms"] = now_ms
        # Log first publish
        if not _tactile_cache.get("first_write_logged"):
            print(f"[inspire_tactile] First tactile data written to DDS shm")
            _tactile_cache["first_write_logged"] = True
    except Exception as e:
        print(f"[inspire_tactile] Failed to publish tactile data: {e}")


def _build_tactile_message(sensor_data: Dict[str, torch.Tensor]) -> Dict[str, list]:
    """Build tactile message dictionary from sensor forces.

    Args:
        sensor_data: Dictionary with keys like 'left_tips', 'left_pads', 'left_palm'
                    Values are tensors of shape (num_envs, num_bodies, 3)

    Returns:
        Dictionary matching inspire_hand_touch IDL field names
    """
    tactile_data = {}

    # Get forces from first environment
    left_tips = sensor_data.get("left_tips")
    left_pads = sensor_data.get("left_pads")
    left_palm = sensor_data.get("left_palm")

    if left_tips is None:
        return tactile_data

    # Process each finger
    for i, finger in enumerate(FINGER_ORDER):
        idl_name = FINGER_MAP[finger]

        # Get force vectors (first env only)
        if i < left_tips.shape[1]:
            tip_force = left_tips[0, i].cpu().numpy()
        else:
            tip_force = np.zeros(3)

        # Tip region (3x3 = 9 taxels)
        tip_grid = force_to_taxel_grid(tip_force, TACTILE_GRIDS["tip"])
        tactile_data[f"{idl_name}_tip_touch"] = flatten_taxel_grid(tip_grid)

        # Nail/top region (12x8 = 96 taxels) - uses same distal force
        nail_grid = force_to_taxel_grid(tip_force, TACTILE_GRIDS["nail"])
        tactile_data[f"{idl_name}_top_touch"] = flatten_taxel_grid(nail_grid)

        # Pad region from intermediate link
        if left_pads is not None and i < left_pads.shape[1]:
            pad_force = left_pads[0, i].cpu().numpy()
        else:
            pad_force = np.zeros(3)

        if finger == "thumb":
            # Thumb has middle region (3x3 = 9) and larger pad (12x8 = 96)
            middle_grid = force_to_taxel_grid(pad_force, TACTILE_GRIDS["thumb_middle"])
            tactile_data[f"{idl_name}_middle_touch"] = flatten_taxel_grid(middle_grid)

            pad_grid = force_to_taxel_grid(pad_force, TACTILE_GRIDS["thumb_pad"])
            tactile_data[f"{idl_name}_palm_touch"] = flatten_taxel_grid(pad_grid)
        else:
            # Other fingers have 10x8 = 80 taxel pad
            pad_grid = force_to_taxel_grid(pad_force, TACTILE_GRIDS["pad"])
            tactile_data[f"{idl_name}_palm_touch"] = flatten_taxel_grid(pad_grid)

    # Palm region (8x14 = 112 taxels)
    if left_palm is not None and left_palm.shape[1] > 0:
        palm_force = left_palm[0, 0].cpu().numpy()
    else:
        palm_force = np.zeros(3)

    palm_grid = force_to_taxel_grid(palm_force, TACTILE_GRIDS["palm"])
    tactile_data["palm_touch"] = flatten_taxel_grid(palm_grid, palm=True)

    return tactile_data


# Export for external use
__all__ = [
    "get_inspire_tactile_state",
    "FINGER_MAP",
    "FINGER_ORDER",
]
