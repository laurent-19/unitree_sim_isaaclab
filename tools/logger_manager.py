# Copyright (c) 2025, Unitree Robotics Co., Ltd. All Rights Reserved.
# License: Apache License, Version 2.0
"""
统一日志管理系统
提供结构化日志记录、性能监控和错误追踪
"""

import logging
import logging.handlers
import sys
import os
import time
import threading
from typing import Dict, Optional, Any
from pathlib import Path
from dataclasses import dataclass, asdict
from enum import Enum

class LogLevel(Enum):
    DEBUG = logging.DEBUG
    INFO = logging.INFO
    WARNING = logging.WARNING
    ERROR = logging.ERROR
    CRITICAL = logging.CRITICAL

class LogCategory(Enum):
    RENDER = "render"
    CONTROL = "control"
    DDS = "dds"
    CAMERA = "camera"
    SIMULATION = "simulation"
    PERFORMANCE = "performance"
    SYSTEM = "system"

@dataclass
class LogContext:
    """日志上下文"""
    category: LogCategory
    operation: str
    start_time: Optional[float] = None
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}

class PerformanceTimer:
    """性能计时器"""

    def __init__(self, name: str, category: LogCategory = LogCategory.PERFORMANCE):
        self.name = name
        self.category = category
        self.start_time = time.perf_counter()
        self.checkpoints = {}

    def checkpoint(self, name: str) -> float:
        """记录检查点"""
        current_time = time.perf_counter()
        elapsed = current_time - self.start_time
        self.checkpoints[name] = elapsed
        return elapsed

    def get_elapsed(self) -> float:
        """获取总耗时"""
        return time.perf_counter() - self.start_time

    def log_performance(self, logger: 'LoggerManager', level: LogLevel = LogLevel.DEBUG):
        """记录性能信息"""
        elapsed = self.get_elapsed()
        checkpoints_str = ", ".join(f"{name}: {time:.3f}s" for name, time in self.checkpoints.items())

        logger.log_performance(
            f"Performance: {self.name} - Total: {elapsed:.3f}s, Checkpoints: {checkpoints_str}",
            category=self.category,
            elapsed=elapsed,
            checkpoints=self.checkpoints
        )

class StructuredFormatter(logging.Formatter):
    """结构化日志格式化器"""

    def format(self, record):
        # 添加默认字段
        if not hasattr(record, 'category'):
            record.category = LogCategory.SYSTEM.value
        if not hasattr(record, 'operation'):
            record.operation = 'unknown'
        if not hasattr(record, 'elapsed'):
            record.elapsed = None
        if not hasattr(record, 'metadata'):
            record.metadata = {}

        # 格式化消息
        message = super().format(record)

        # 添加结构化信息
        structured_info = {
            'timestamp': record.created,
            'level': record.levelname,
            'category': record.category,
            'operation': record.operation,
            'message': record.getMessage(),
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno
        }

        if record.elapsed is not None:
            structured_info['elapsed'] = record.elapsed
        if record.metadata:
            structured_info.update(record.metadata)

        return f"{message} | {structured_info}"

class LoggerManager:
    """日志管理器"""

    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super(LoggerManager, cls).__new__(cls)
        return cls._instance

    def __init__(self):
        if hasattr(self, '_initialized'):
            return

        self._initialized = True
        self._loggers = {}
        self._default_level = LogLevel.INFO
        self._log_dir = Path("logs")
        self._log_dir.mkdir(exist_ok=True)

        # 创建根日志器
        self._setup_root_logger()

        # 性能统计
        self._performance_stats = {}
        self._error_counts = {}

    def _setup_root_logger(self):
        """设置根日志器"""
        root_logger = logging.getLogger('unitree_sim')
        root_logger.setLevel(self._default_level.value)

        # 控制台处理器
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(self._default_level.value)
        console_formatter = StructuredFormatter(
            '%(asctime)s [%(levelname)s] %(category)s:%(operation)s - %(message)s'
        )
        console_handler.setFormatter(console_formatter)
        root_logger.addHandler(console_handler)

        # 文件处理器（按日期轮转）
        file_handler = logging.handlers.TimedRotatingFileHandler(
            self._log_dir / "unitree_sim.log",
            when='midnight',
            interval=1,
            backupCount=30
        )
        file_handler.setLevel(LogLevel.DEBUG.value)
        file_formatter = StructuredFormatter(
            '%(asctime)s [%(levelname)s] %(category)s:%(operation)s - %(message)s'
        )
        file_handler.setFormatter(file_formatter)
        root_logger.addHandler(file_handler)

        self._root_logger = root_logger

    def get_logger(self, name: str = "unitree_sim") -> logging.Logger:
        """获取日志器"""
        if name not in self._loggers:
            logger = logging.getLogger(f"unitree_sim.{name}")
            logger.setLevel(self._default_level.value)
            self._loggers[name] = logger
        return self._loggers[name]

    def set_level(self, level: LogLevel, logger_name: Optional[str] = None):
        """设置日志级别"""
        if logger_name:
            logger = self.get_logger(logger_name)
            logger.setLevel(level.value)
        else:
            self._default_level = level
            self._root_logger.setLevel(level.value)

    def log(self, level: LogLevel, message: str,
            category: LogCategory = LogCategory.SYSTEM,
            operation: str = "unknown",
            elapsed: Optional[float] = None,
            metadata: Optional[Dict[str, Any]] = None,
            logger_name: Optional[str] = None):
        """记录日志"""
        logger = self.get_logger(logger_name or "main")

        # 创建日志记录
        record = logger.makeRecord(
            logger.name, level.value, "(unknown file)", 0, message, (), None
        )

        # 添加自定义字段
        record.category = category.value
        record.operation = operation
        record.elapsed = elapsed
        record.metadata = metadata or {}

        # 记录日志
        logger.handle(record)

        # 更新统计
        if level == LogLevel.ERROR:
            error_key = f"{category.value}:{operation}"
            self._error_counts[error_key] = self._error_counts.get(error_key, 0) + 1

    def log_performance(self, message: str,
                       category: LogCategory = LogCategory.PERFORMANCE,
                       elapsed: Optional[float] = None,
                       checkpoints: Optional[Dict[str, float]] = None,
                       metadata: Optional[Dict[str, Any]] = None):
        """记录性能日志"""
        metadata = metadata or {}
        if checkpoints:
            metadata['checkpoints'] = checkpoints

        perf_key = f"{category.value}:{message.split(' - ')[0]}"
        self._performance_stats[perf_key] = {
            'elapsed': elapsed,
            'timestamp': time.time(),
            'metadata': metadata
        }

        self.log(LogLevel.DEBUG, message, category, "performance",
                elapsed, metadata)

    def log_error(self, error: Exception, operation: str,
                 category: LogCategory = LogCategory.SYSTEM,
                 metadata: Optional[Dict[str, Any]] = None):
        """记录错误日志"""
        metadata = metadata or {}
        metadata['exception_type'] = type(error).__name__
        metadata['exception_args'] = error.args

        self.log(LogLevel.ERROR,
                f"Exception in {operation}: {str(error)}",
                category, operation, metadata=metadata)

    def start_operation(self, operation: str, category: LogCategory) -> LogContext:
        """开始操作跟踪"""
        return LogContext(
            category=category,
            operation=operation,
            start_time=time.perf_counter()
        )

    def end_operation(self, context: LogContext, level: LogLevel = LogLevel.DEBUG,
                     metadata: Optional[Dict[str, Any]] = None):
        """结束操作跟踪"""
        if context.start_time is None:
            return

        elapsed = time.perf_counter() - context.start_time
        metadata = metadata or {}
        metadata.update(context.metadata)

        self.log(level, f"Completed {context.operation}",
                context.category, context.operation, elapsed, metadata)

    def get_performance_stats(self) -> Dict[str, Any]:
        """获取性能统计"""
        return {
            'performance_stats': self._performance_stats.copy(),
            'error_counts': self._error_counts.copy(),
            'total_errors': sum(self._error_counts.values())
        }

    def create_timer(self, name: str, category: LogCategory = LogCategory.PERFORMANCE) -> PerformanceTimer:
        """创建性能计时器"""
        return PerformanceTimer(name, category)

# 全局日志管理器实例
logger_manager = LoggerManager()

# 便捷函数
def get_logger(name: str = "main") -> logging.Logger:
    """获取日志器"""
    return logger_manager.get_logger(name)

def log_performance(message: str, **kwargs):
    """记录性能日志"""
    logger_manager.log_performance(message, **kwargs)

def log_error(error: Exception, operation: str, **kwargs):
    """记录错误日志"""
    logger_manager.log_error(error, operation, **kwargs)
