# Copyright (c) 2025, Unitree Robotics Co., Ltd. All Rights Reserved.
# License: Apache License, Version 2.0
"""
错误处理工具
提供基本的错误处理和日志记录功能
"""

from tools.logger_manager import get_logger

logger = get_logger("errors")

class UnitreeSimError(Exception):
    """基础异常类"""
    pass

def handle_error(error: Exception, component: str = "unknown", operation: str = "unknown"):
    """处理错误并记录日志"""
    logger.error(f"Error in {component}:{operation}: {str(error)}", extra={
        'error_type': type(error).__name__,
        'component': component,
        'operation': operation
    })

def safe_execute(func, fallback=None, log_error=True):
    """安全执行函数"""
    try:
        return func()
    except Exception as e:
        if log_error:
            logger.warning(f"Safe execute failed: {e}")
        return fallback
