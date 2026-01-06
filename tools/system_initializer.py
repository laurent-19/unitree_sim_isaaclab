# Copyright (c) 2025, Unitree Robotics Co., Ltd. All Rights Reserved.
# License: Apache License, Version 2.0
"""
系统初始化工具
提供基本的系统启动和清理功能
"""

import os
import signal
import atexit
from typing import Dict, Any, Optional
from tools.logger_manager import get_logger

logger = get_logger("system")

# 全局关闭标志
_shutdown_requested = False

def init_system(config_overrides: Optional[Dict[str, Any]] = None) -> bool:
    """系统初始化"""
    try:
        # 设置项目根目录
        if 'PROJECT_ROOT' not in os.environ:
            project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            os.environ['PROJECT_ROOT'] = project_root

        # 加载配置
        from tools.config_manager import config_manager
        config_manager.load_config()

        # 应用配置覆盖
        if config_overrides:
            for section, values in config_overrides.items():
                for key, value in values.items():
                    config_manager.set_config_value(section, key, value)

        # 启动性能监控
        from tools.performance_monitor import performance_monitor
        performance_monitor.start_monitoring()

        # 注册信号处理器
        def signal_handler(signum, frame):
            global _shutdown_requested
            logger.warning(f"Received signal {signum}, shutting down")
            _shutdown_requested = True
            cleanup_system()

        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
        atexit.register(cleanup_system)

        logger.info("System initialization completed")
        return True

    except Exception as e:
        logger.error(f"System initialization failed: {e}")
        return False


def cleanup_system():
    """系统清理"""
    try:
        from tools.performance_monitor import performance_monitor
        from tools.resource_manager import resource_manager

        performance_monitor.stop_monitoring()
        resource_manager.cleanup_all()
        logger.info("System cleanup completed")
    except Exception as e:
        logger.error(f"System cleanup failed: {e}")


def is_shutdown_requested() -> bool:
    """检查是否请求了关闭"""
    global _shutdown_requested
    return _shutdown_requested
