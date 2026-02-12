# Copyright (c) 2025, Unitree Robotics Co., Ltd. All Rights Reserved.
# License: Apache License, Version 2.0
"""
TacSL Visuo-Tactile Sensor Extraction Module.

Extracts force field data from TacSL VisuoTactileSensor instances and
converts them to taxel arrays matching the Inspire hand DDS format.

Unlike ContactSensor which uses Gaussian force-to-taxel approximation,
TacSL provides per-taxel forces directly from the force field simulation.

Usage:
    from tasks.common_observations.tacsl_tactile import get_tacsl_tactile_state

    # In observation function
    tactile_obs = get_tacsl_tactile_state(env, enable_dds=True)
"""

from __future__ import annotations

import torch
import numpy as np
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple, Any

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

# Try to import TacSL (Isaac Lab v2.3.2+)
try:
    from isaaclab_contrib.sensors.tacsl_sensor import VisuoTactileSensor
    TACSL_AVAILABLE = True
except ImportError:
    try:
        from isaaclab_contrib.sensors import VisuoTactileSensor
        TACSL_AVAILABLE = True
    except ImportError:
        VisuoTactileSensor = None
        TACSL_AVAILABLE = False


# Sensor name prefixes for TacSL sensors in scene
TACSL_SENSOR_PREFIX = "tacsl_"

# Finger ordering matching IDL format
FINGER_ORDER = ["pinky", "ring", "middle", "index", "thumb"]

# Regions per finger (order matters for concatenation)
FINGER_REGIONS = ["tip", "nail", "pad"]
THUMB_REGIONS = ["tip", "nail", "middle", "pad"]

# IDL field name mapping
FINGER_TO_IDL = {
    "pinky": "fingerone",
    "ring": "fingertwo",
    "middle": "fingerthree",
    "index": "fingerfour",
    "thumb": "fingerfive",
}

# Module-level cache for TacSL state
_tacsl_cache = {
    "sensors_detected": None,
    "sensor_names": {},
    "dds_instances": {"l": None, "r": None},
    "dds_initialized": False,
    "last_publish_ms": 0,
    "publish_interval_ms": 20,  # 50Hz
}


def is_tacsl_available() -> bool:
    """Check if TacSL module is available."""
    return TACSL_AVAILABLE


def detect_tacsl_sensors(env: ManagerBasedRLEnv) -> Dict[str, List[str]]:
    """Detect available TacSL sensors in the environment.

    Args:
        env: Isaac Lab environment instance

    Returns:
        Dict with keys "L" and "R" containing lists of sensor names
    """
    global _tacsl_cache

    if _tacsl_cache["sensors_detected"] is not None:
        return _tacsl_cache["sensor_names"]

    sensors = {"L": [], "R": []}

    if not TACSL_AVAILABLE:
        _tacsl_cache["sensors_detected"] = False
        return sensors

    # Look for TacSL sensors in scene
    for sensor_name in env.scene.sensors:
        if not sensor_name.startswith(TACSL_SENSOR_PREFIX):
            continue

        sensor = env.scene[sensor_name]
        if not isinstance(sensor, VisuoTactileSensor):
            continue

        # Parse sensor name: tacsl_{side}_{finger}_{region}
        parts = sensor_name[len(TACSL_SENSOR_PREFIX):].split("_")
        if len(parts) >= 2:
            side = parts[0]
            if side in ["L", "R"]:
                sensors[side].append(sensor_name)

    _tacsl_cache["sensor_names"] = sensors
    _tacsl_cache["sensors_detected"] = any(sensors["L"]) or any(sensors["R"])

    if _tacsl_cache["sensors_detected"]:
        print(f"[tacsl_tactile] Detected TacSL sensors: L={len(sensors['L'])}, R={len(sensors['R'])}")

    return sensors


def has_tacsl_sensors(env: ManagerBasedRLEnv) -> bool:
    """Check if environment has TacSL sensors configured.

    Args:
        env: Isaac Lab environment instance

    Returns:
        True if TacSL sensors are present
    """
    if not TACSL_AVAILABLE:
        return False

    sensors = detect_tacsl_sensors(env)
    return bool(sensors["L"]) or bool(sensors["R"])


def _get_tacsl_dds_instances():
    """Get DDS instances for TacSL tactile publishing."""
    global _tacsl_cache
    import sys
    import os

    if _tacsl_cache["dds"]["l"] is None or _tacsl_cache["dds"]["r"] is None:
        try:
            sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), 'dds'))
            from dds.dds_master import dds_manager
            _tacsl_cache["dds_instances"]["l"] = dds_manager.get_object("inspire_touch_l")
            _tacsl_cache["dds_instances"]["r"] = dds_manager.get_object("inspire_touch_r")

            if _tacsl_cache["dds_instances"]["l"] or _tacsl_cache["dds_instances"]["r"]:
                print(f"[tacsl_tactile] DDS instances: L={_tacsl_cache['dds_instances']['l'] is not None}, "
                      f"R={_tacsl_cache['dds_instances']['r'] is not None}")
            _tacsl_cache["dds_initialized"] = True
        except Exception as e:
            if not _tacsl_cache["dds_initialized"]:
                print(f"[tacsl_tactile] DDS init failed: {e}")
                _tacsl_cache["dds_initialized"] = True

    return _tacsl_cache["dds_instances"]


def extract_tacsl_force_field(
    env: ManagerBasedRLEnv,
    sensor_name: str,
) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
    """Extract normal and shear force fields from a TacSL sensor.

    Args:
        env: Isaac Lab environment instance
        sensor_name: Name of the TacSL sensor in scene

    Returns:
        Tuple of (normal_forces, shear_forces):
            - normal_forces: Shape (num_envs, H, W) - normal force per taxel
            - shear_forces: Shape (num_envs, H, W, 2) - shear forces (x, y) per taxel
        Returns (None, None) if sensor not found or TacSL not available
    """
    if not TACSL_AVAILABLE:
        return None, None

    if sensor_name not in env.scene.sensors:
        return None, None

    sensor = env.scene[sensor_name]
    if not isinstance(sensor, VisuoTactileSensor):
        return None, None

    try:
        # TacSL sensor data access
        # tactile_normal_force: (num_envs, H, W)
        # tactile_shear_force: (num_envs, H, W, 2)
        normal = sensor.data.tactile_normal_force
        shear = sensor.data.tactile_shear_force
        return normal, shear
    except AttributeError:
        # Handle case where sensor doesn't have expected attributes
        return None, None


def tacsl_to_taxel_array(
    normal_force: torch.Tensor,
    scale: float = 1000.0,
    max_value: int = 4095,
) -> np.ndarray:
    """Convert TacSL normal force field to taxel array.

    Unlike ContactSensor Gaussian approximation, TacSL provides
    per-taxel forces directly - just need scaling and clamping.

    Args:
        normal_force: Force field tensor of shape (H, W)
        scale: Conversion factor from Newtons to taxel units
        max_value: Maximum taxel value (12-bit = 4095)

    Returns:
        2D numpy array of int16 taxel values
    """
    # Scale forces to taxel units
    scaled = (normal_force * scale).cpu().numpy()

    # Clip to valid range and convert to int16
    taxels = np.clip(scaled, 0, max_value).astype(np.int16)

    return taxels


def get_tacsl_tactile_state(
    env: ManagerBasedRLEnv,
    enable_dds: bool = True,
) -> torch.Tensor:
    """Extract tactile data from TacSL sensors.

    This function reads force fields from TacSL VisuoTactileSensor instances
    and converts them to flattened taxel arrays for RL observations.

    Args:
        env: Isaac Lab ManagerBasedRLEnv instance
        enable_dds: Whether to publish tactile data via DDS

    Returns:
        torch.Tensor: Flattened tactile forces for all sensors
                     Shape: (num_envs, total_taxels)
    """
    device = env.device
    batch = env.num_envs

    if not TACSL_AVAILABLE:
        return torch.zeros(batch, 1, device=device)

    sensors = detect_tacsl_sensors(env)
    if not _tacsl_cache["sensors_detected"]:
        return torch.zeros(batch, 1, device=device)

    # Collect force tensors from all TacSL sensors
    force_tensors = []
    sensor_data = {"L": {}, "R": {}}

    for side in ["L", "R"]:
        for sensor_name in sensors[side]:
            normal, shear = extract_tacsl_force_field(env, sensor_name)
            if normal is not None:
                # Flatten force field for RL observation
                flat_normal = normal.reshape(batch, -1)
                force_tensors.append(flat_normal)

                # Store for DDS publishing
                region_name = sensor_name[len(TACSL_SENSOR_PREFIX) + 2:]  # Remove "tacsl_X_"
                sensor_data[side][region_name] = (normal, shear)

    if not force_tensors:
        return torch.zeros(batch, 1, device=device)

    # Publish to DDS for first environment
    if enable_dds and batch > 0:
        _publish_tacsl_to_dds(sensor_data)

    # Concatenate all forces for RL observation
    all_forces = torch.cat(force_tensors, dim=-1)
    return all_forces


def _publish_tacsl_to_dds(sensor_data: Dict[str, Dict[str, Tuple[torch.Tensor, torch.Tensor]]]):
    """Publish TacSL tactile data to DDS.

    Args:
        sensor_data: Dict mapping side ("L"/"R") to dict of region data
    """
    import time
    global _tacsl_cache

    now_ms = int(time.time() * 1000)
    if now_ms - _tacsl_cache["last_publish_ms"] < _tacsl_cache["publish_interval_ms"]:
        return

    dds = _get_tacsl_dds_instances()
    if not dds["l"] and not dds["r"]:
        return

    try:
        # Publish left hand
        if dds["l"] and sensor_data["L"]:
            left_msg = _build_tacsl_tactile_message(sensor_data["L"])
            dds["l"].write_tactile_data(left_msg)

        # Publish right hand
        if dds["r"] and sensor_data["R"]:
            right_msg = _build_tacsl_tactile_message(sensor_data["R"])
            dds["r"].write_tactile_data(right_msg)

        _tacsl_cache["last_publish_ms"] = now_ms

    except Exception as e:
        print(f"[tacsl_tactile] DDS publish failed: {e}")


def _build_tacsl_tactile_message(
    region_data: Dict[str, Tuple[torch.Tensor, torch.Tensor]],
) -> Dict[str, list]:
    """Build tactile message from TacSL sensor data.

    Args:
        region_data: Dict mapping region names to (normal, shear) tensor pairs

    Returns:
        Dict matching inspire_hand_touch IDL field format
    """
    tactile_data = {}

    for finger in FINGER_ORDER:
        idl_name = FINGER_TO_IDL[finger]
        regions = THUMB_REGIONS if finger == "thumb" else FINGER_REGIONS

        for region in regions:
            sensor_key = f"{finger}_{region}"
            if sensor_key in region_data:
                normal, _ = region_data[sensor_key]
                # Extract first environment
                taxels = tacsl_to_taxel_array(normal[0])

                # Determine IDL field name
                if region == "tip":
                    field = f"{idl_name}_tip_touch"
                elif region == "nail":
                    field = f"{idl_name}_top_touch"
                elif region == "middle":
                    field = f"{idl_name}_middle_touch"
                else:  # pad
                    field = f"{idl_name}_palm_touch"

                tactile_data[field] = taxels.flatten().tolist()

    # Handle palm region
    if "palm" in region_data:
        normal, _ = region_data["palm"]
        taxels = tacsl_to_taxel_array(normal[0])
        # Palm uses column-major from bottom row
        tactile_data["palm_touch"] = taxels[::-1, :].T.flatten().tolist()

    return tactile_data


# Export public API
__all__ = [
    "TACSL_AVAILABLE",
    "is_tacsl_available",
    "has_tacsl_sensors",
    "detect_tacsl_sensors",
    "extract_tacsl_force_field",
    "tacsl_to_taxel_array",
    "get_tacsl_tactile_state",
]
