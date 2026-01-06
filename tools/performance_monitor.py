# Copyright (c) 2025, Unitree Robotics Co., Ltd. All Rights Reserved.
# License: Apache License, Version 2.0
"""
性能监控工具
提供基本的性能监控功能
"""

import time
import psutil
from tools.logger_manager import get_logger

logger = get_logger("performance")

class PerformanceMonitor:
    """简单的性能监控器"""

    def __init__(self):
        self.monitoring = False
        self.start_time = None

    def start_monitoring(self):
        """开始监控"""
        self.monitoring = True
        self.start_time = time.time()
        logger.info("Performance monitoring started")

    def stop_monitoring(self):
        """停止监控"""
        self.monitoring = False
        if self.start_time:
            duration = time.time() - self.start_time
            logger.info(f"Performance monitoring stopped after {duration:.1f}s")
        self.start_time = None

    def record_operation_time(self, operation: str, duration: float):
        """记录操作耗时"""
        if self.monitoring:
            logger.debug(f"Operation {operation} took {duration*1000:.1f}ms")

    def start_benchmark(self, name: str):
        """开始基准测试"""
        return f"benchmark_{int(time.time())}"

    def end_benchmark(self):
        """结束基准测试"""
        return None

    def get_summary_stats(self):
        """获取统计信息"""
        try:
            cpu_percent = psutil.cpu_percent(interval=None)
            memory_percent = psutil.virtual_memory().percent
            return {
                'cpu_percent': {'current': cpu_percent},
                'memory_percent': {'current': memory_percent}
            }
        except Exception as e:
            logger.warning(f"Failed to get system stats: {e}")
            return {}

# 全局性能监控器实例
performance_monitor = PerformanceMonitor()