"""
公共配置模块
提供可复用的机器人和相机配置
"""

from .robot_configs import RobotBaseCfg, H12RobotPresets, RobotJointTemplates, G1RobotPresets
from .camera_configs import CameraBaseCfg, CameraPresets
from .tactile_configs import (
    TACSL_AVAILABLE,
    TactileRegionSpec,
    TACTILE_REGION_SPECS,
    FINGER_TO_IDL,
    TactileSensorBaseCfg,
    InspireHandTactilePresets,
)

__all__ = [
    # Robot configs
    "RobotBaseCfg",
    "G1RobotPresets",
    "H12RobotPresets",
    "RobotJointTemplates",
    # Camera configs
    "CameraBaseCfg",
    "CameraPresets",
    # Tactile configs
    "TACSL_AVAILABLE",
    "TactileRegionSpec",
    "TACTILE_REGION_SPECS",
    "FINGER_TO_IDL",
    "TactileSensorBaseCfg",
    "InspireHandTactilePresets",
] 