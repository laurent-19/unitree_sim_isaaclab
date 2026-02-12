# Copyright (c) 2025, Unitree Robotics Co., Ltd. All Rights Reserved.
# License: Apache License, Version 2.0
"""
Inspire Hand tactile state extraction and DDS publishing.

Supports two backends:
1. ContactSensor: Uses Gaussian force-to-taxel approximation (default fallback)
2. TacSL: Per-taxel force computation from visuo-tactile force field simulation

The backend is auto-detected based on available sensors in the environment.
TacSL provides higher fidelity sim-to-real transfer but requires:
- isaaclab_contrib package installed
- USD assets with elastomer geometry
- TacSL sensor configs in scene

Usage:
    # Auto-detect backend (recommended)
    tactile = get_inspire_tactile_state(env, enable_dds=True)

    # Force specific backend
    tactile = get_inspire_tactile_state(env, enable_dds=True, backend="tacsl")
    tactile = get_inspire_tactile_state(env, enable_dds=True, backend="contact")
"""

from __future__ import annotations

import torch
import numpy as np
import sys
import os
from typing import TYPE_CHECKING, Optional, Dict, Any, Literal

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

from .tactile_mapping import (
    force_to_taxel_grid,
    flatten_taxel_grid,
    TACTILE_GRIDS,
)

# Import TacSL module (may not be available)
try:
    from .tacsl_tactile import (
        TACSL_AVAILABLE,
        has_tacsl_sensors,
        get_tacsl_tactile_state as _get_tacsl_state,
    )
except ImportError:
    TACSL_AVAILABLE = False
    has_tacsl_sensors = lambda env: False
    _get_tacsl_state = None

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
    "dds": {"l": None, "r": None},
    "dds_initialized": False,
    "last_publish_ms": 0,
    "publish_interval_ms": 20,  # 50Hz publishing rate
    "backend": None,  # Auto-detected backend: "tacsl" or "contact"
    "backend_checked": False,
}


def _get_touch_dds_instances():
    """Get the inspire_touch DDS instances for both hands with lazy initialization."""
    global _tactile_cache

    # Always try to get DDS if we don't have them yet
    if _tactile_cache["dds"]["l"] is None or _tactile_cache["dds"]["r"] is None:
        try:
            sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), 'dds'))
            from dds.dds_master import dds_manager
            _tactile_cache["dds"]["l"] = dds_manager.get_object("inspire_touch_l")
            _tactile_cache["dds"]["r"] = dds_manager.get_object("inspire_touch_r")
            if _tactile_cache["dds"]["l"] or _tactile_cache["dds"]["r"]:
                print(f"[inspire_tactile] DDS touch instances obtained: L={_tactile_cache['dds']['l'] is not None}, R={_tactile_cache['dds']['r'] is not None}")
            # Only log failure once
            elif not _tactile_cache["dds_initialized"]:
                print("[inspire_tactile] DDS touch instances not found (will retry)")
        except Exception as e:
            if not _tactile_cache["dds_initialized"]:
                print(f"[inspire_tactile] DDS init failed: {e}")
        _tactile_cache["dds_initialized"] = True

    return _tactile_cache["dds"]


def _check_sensor_available(env: ManagerBasedRLEnv, sensor_name: str) -> bool:
    """Check if a sensor is available in the scene."""
    return sensor_name in env.scene.sensors


def _detect_backend(env: ManagerBasedRLEnv) -> str:
    """Auto-detect the best available tactile backend.

    Args:
        env: Isaac Lab environment instance

    Returns:
        "tacsl" if TacSL sensors available, "contact" otherwise
    """
    global _tactile_cache

    if _tactile_cache["backend_checked"]:
        return _tactile_cache["backend"]

    # Check for TacSL sensors first (higher fidelity)
    if TACSL_AVAILABLE and has_tacsl_sensors(env):
        _tactile_cache["backend"] = "tacsl"
        print("[inspire_tactile] Using TacSL backend (high-fidelity force fields)")
    else:
        _tactile_cache["backend"] = "contact"
        if not TACSL_AVAILABLE:
            print("[inspire_tactile] Using ContactSensor backend (TacSL not installed)")
        else:
            print("[inspire_tactile] Using ContactSensor backend (no TacSL sensors in scene)")

    _tactile_cache["backend_checked"] = True
    return _tactile_cache["backend"]


def _should_use_tacsl(env: ManagerBasedRLEnv, backend: str) -> bool:
    """Determine if TacSL backend should be used.

    Args:
        env: Isaac Lab environment instance
        backend: Requested backend ("auto", "tacsl", or "contact")

    Returns:
        True if TacSL should be used
    """
    if backend == "contact":
        return False
    if backend == "tacsl":
        if not TACSL_AVAILABLE:
            print("[inspire_tactile] Warning: TacSL requested but not available, falling back to ContactSensor")
            return False
        return True
    # Auto-detect
    return _detect_backend(env) == "tacsl"


def get_inspire_tactile_state(
    env: ManagerBasedRLEnv,
    enable_dds: bool = True,
    backend: Literal["auto", "contact", "tacsl"] = "auto",
) -> torch.Tensor:
    """Extract tactile data from sensors and optionally publish to DDS.

    Supports two backends:
    - "contact": ContactSensor with Gaussian force-to-taxel mapping
    - "tacsl": TacSL visuo-tactile force field sensors (higher fidelity)

    The backend is auto-detected by default, preferring TacSL when available.

    Args:
        env: Isaac Lab ManagerBasedRLEnv instance
        enable_dds: Whether to publish tactile data via DDS
        backend: Backend selection ("auto", "contact", or "tacsl")

    Returns:
        torch.Tensor: Flattened contact forces for all finger sensors
                     Shape: (num_envs, total_force_components)
    """
    # Use TacSL backend if available and selected
    if _should_use_tacsl(env, backend):
        return _get_tacsl_state(env, enable_dds=enable_dds)

    # Fall back to ContactSensor backend
    return _get_contact_tactile_state(env, enable_dds=enable_dds)


def _get_contact_tactile_state(
    env: ManagerBasedRLEnv,
    enable_dds: bool = True,
) -> torch.Tensor:
    """Extract tactile data from ContactSensor (original implementation).

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

    # Publish to DDS for first environment (both left and right hands)
    if enable_dds and batch > 0:
        has_left = "left_tips" in sensor_data
        has_right = "right_tips" in sensor_data
        if has_left or has_right:
            _publish_tactile_to_dds(sensor_data)
        elif not _tactile_cache.get("no_data_logged"):
            # Log once why publishing isn't happening
            print(f"[inspire_tactile] Not publishing: no sensor data, sensors={list(sensor_data.keys())}")
            _tactile_cache["no_data_logged"] = True

    # Concatenate all forces for RL observation
    all_forces = torch.cat(force_tensors, dim=-1)
    return all_forces


def _publish_tactile_to_dds(sensor_data: Dict[str, torch.Tensor]):
    """Publish tactile data to DDS with rate limiting for both hands.

    Args:
        sensor_data: Dictionary mapping sensor names to force tensors
    """
    import time
    global _tactile_cache

    now_ms = int(time.time() * 1000)
    if now_ms - _tactile_cache["last_publish_ms"] < _tactile_cache["publish_interval_ms"]:
        return

    dds_instances = _get_touch_dds_instances()
    if not dds_instances["l"] and not dds_instances["r"]:
        if not _tactile_cache.get("no_dds_logged"):
            print("[inspire_tactile] DDS objects not available for publishing")
            _tactile_cache["no_dds_logged"] = True
        return

    try:
        # Publish left hand tactile data
        if dds_instances["l"] and "left_tips" in sensor_data:
            left_data = _build_tactile_message(sensor_data, side="left")
            dds_instances["l"].write_tactile_data(left_data)

        # Publish right hand tactile data
        if dds_instances["r"] and "right_tips" in sensor_data:
            right_data = _build_tactile_message(sensor_data, side="right")
            dds_instances["r"].write_tactile_data(right_data)

        _tactile_cache["last_publish_ms"] = now_ms
        # Log first publish
        if not _tactile_cache.get("first_write_logged"):
            print(f"[inspire_tactile] First tactile data written to DDS shm")
            _tactile_cache["first_write_logged"] = True
    except Exception as e:
        print(f"[inspire_tactile] Failed to publish tactile data: {e}")


def _build_tactile_message(sensor_data: Dict[str, torch.Tensor], side: str = "left") -> Dict[str, list]:
    """Build tactile message dictionary from sensor forces.

    Args:
        sensor_data: Dictionary with keys like '{side}_tips', '{side}_pads', '{side}_palm'
                    Values are tensors of shape (num_envs, num_bodies, 3)
        side: Which hand to build the message for ('left' or 'right')

    Returns:
        Dictionary matching inspire_hand_touch IDL field names
    """
    tactile_data = {}

    # Get forces from first environment for the specified side
    tips = sensor_data.get(f"{side}_tips")
    pads = sensor_data.get(f"{side}_pads")
    palm = sensor_data.get(f"{side}_palm")

    if tips is None:
        return tactile_data

    # Process each finger
    for i, finger in enumerate(FINGER_ORDER):
        idl_name = FINGER_MAP[finger]

        # Get force vectors (first env only)
        if i < tips.shape[1]:
            tip_force = tips[0, i].cpu().numpy()
        else:
            tip_force = np.zeros(3)

        # Tip region (3x3 = 9 taxels)
        tip_grid = force_to_taxel_grid(tip_force, TACTILE_GRIDS["tip"])
        tactile_data[f"{idl_name}_tip_touch"] = flatten_taxel_grid(tip_grid)

        # Nail/top region (12x8 = 96 taxels) - uses same distal force
        nail_grid = force_to_taxel_grid(tip_force, TACTILE_GRIDS["nail"])
        tactile_data[f"{idl_name}_top_touch"] = flatten_taxel_grid(nail_grid)

        # Pad region from intermediate link
        if pads is not None and i < pads.shape[1]:
            pad_force = pads[0, i].cpu().numpy()
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
    if palm is not None and palm.shape[1] > 0:
        palm_force = palm[0, 0].cpu().numpy()
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
    "TACSL_AVAILABLE",
]
