# Copyright (c) 2025, Unitree Robotics Co., Ltd. All Rights Reserved.
# License: Apache License, Version 2.0
"""
配置管理工具
提供基本的配置管理功能
"""

from enum import Enum

class TaskType(Enum):
    EVALUATION = "evaluation"
    REAL_TIME = "real_time"
    REPLAY = "replay"
    DEFAULT = "default"

class RobotType(Enum):
    G129 = "g129"
    H1_2 = "h1_2"

class ConfigManager:
    """简单的配置管理器"""

    def __init__(self):
        self._config = {}

    def load_config(self):
        """加载配置"""
        pass

    def set_config_value(self, section: str, key: str, value):
        """设置配置值"""
        if section not in self._config:
            self._config[section] = {}
        self._config[section][key] = value

    def get_config_value(self, section: str, key: str):
        """获取配置值"""
        return self._config.get(section, {}).get(key)

    def get_config(self):
        """获取完整配置"""
        return self._config

# 全局配置管理器实例
config_manager = ConfigManager()