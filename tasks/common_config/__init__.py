"""
公共配置模块
提供可复用的机器人和相机配置
"""

from .robot_configs import RobotBaseCfg, H12RobotPresets, RobotJointTemplates,G1RobotPresets
from .camera_configs import CameraBaseCfg, CameraPresets
from .contact_sensor_configs import InspireHandContactSensorCfg, INSPIRE_FINGER_BODY_NAMES

__all__ = [
    "RobotBaseCfg",
    "G1RobotPresets",
    "H12RobotPresets",
    "RobotJointTemplates", 
    "CameraBaseCfg",
    "CameraPresets",
    "InspireHandContactSensorCfg",
    "INSPIRE_FINGER_BODY_NAMES",
] 