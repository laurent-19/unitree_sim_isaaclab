# Copyright (c) 2025, Unitree Robotics Co., Ltd. All Rights Reserved.
# License: Apache License, Version 2.0
"""
智能渲染优化器
根据系统负载和任务需求动态调整渲染参数
"""

import time
import psutil
import threading
from typing import Dict, Optional
from dataclasses import dataclass
from enum import Enum

class TaskType(Enum):
    EVALUATION = "evaluation"
    REAL_TIME = "real_time"
    REPLAY = "replay"

@dataclass
class RenderConfig:
    """渲染配置"""
    render_interval: int
    camera_update_period: float
    dds_publish_rate: float
    enable_jpeg: bool
    jpeg_quality: int

class RenderOptimizer:
    """智能渲染优化器"""

    def __init__(self, task_type: TaskType = TaskType.REAL_TIME):
        self.task_type = task_type
        self.system_load_history = []
        self.config_history = []
        self.lock = threading.Lock()

        # 性能监控
        self.cpu_percent = 0.0
        self.memory_percent = 0.0
        self.last_update = time.time()

        # 优化配置映射
        self._config_presets = {
            TaskType.EVALUATION: RenderConfig(
                render_interval=1, camera_update_period=0.02,
                dds_publish_rate=100.0, enable_jpeg=True, jpeg_quality=90
            ),
            TaskType.REAL_TIME: RenderConfig(
                render_interval=1, camera_update_period=0.02,
                dds_publish_rate=100.0, enable_jpeg=False, jpeg_quality=95
            ),
            TaskType.REPLAY: RenderConfig(
                render_interval=5, camera_update_period=0.1,
                dds_publish_rate=20.0, enable_jpeg=True, jpeg_quality=85
            )
        }

        # 启动监控线程
        self.monitoring_thread = threading.Thread(target=self._monitor_system_load, daemon=True)
        self.monitoring_thread.start()

    def get_optimal_config(self) -> RenderConfig:
        """获取当前最优渲染配置"""
        with self.lock:
            # 基于系统负载动态调整
            base_config = self._config_presets[self.task_type]

            if self.cpu_percent > 80 or self.memory_percent > 85:
                # 高负载时降低性能要求
                return RenderConfig(
                    render_interval=max(base_config.render_interval * 2, 10),
                    camera_update_period=base_config.camera_update_period * 2,
                    dds_publish_rate=max(base_config.dds_publish_rate * 0.5, 10.0),
                    enable_jpeg=True,
                    jpeg_quality=max(base_config.jpeg_quality - 10, 70)
                )
            elif self.cpu_percent < 30 and self.memory_percent < 50:
                # 低负载时提高质量
                return RenderConfig(
                    render_interval=max(base_config.render_interval // 2, 1),
                    camera_update_period=base_config.camera_update_period * 0.8,
                    dds_publish_rate=min(base_config.dds_publish_rate * 1.2, 500.0),
                    enable_jpeg=base_config.enable_jpeg,
                    jpeg_quality=min(base_config.jpeg_quality + 5, 100)
                )
            else:
                return base_config

    def _monitor_system_load(self):
        """监控系统负载"""
        while True:
            try:
                self.cpu_percent = psutil.cpu_percent(interval=1)
                self.memory_percent = psutil.virtual_memory().percent
                self.last_update = time.time()

                # 保持最近10次的负载历史
                with self.lock:
                    self.system_load_history.append((self.cpu_percent, self.memory_percent))
                    if len(self.system_load_history) > 10:
                        self.system_load_history.pop(0)

            except Exception as e:
                print(f"[RenderOptimizer] Monitoring error: {e}")

            time.sleep(2)  # 每2秒更新一次

    def get_system_status(self) -> Dict:
        """获取系统状态信息"""
        return {
            "cpu_percent": self.cpu_percent,
            "memory_percent": self.memory_percent,
            "task_type": self.task_type.value,
            "last_update": self.last_update
        }
