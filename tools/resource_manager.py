# Copyright (c) 2025, Unitree Robotics Co., Ltd. All Rights Reserved.
# License: Apache License, Version 2.0
"""
资源管理工具
提供基本的资源清理功能
"""

import gc
from tools.logger_manager import get_logger

logger = get_logger("resource")

class ResourceManager:
    """简单的资源管理器"""

    def cleanup_all(self):
        """清理所有资源"""
        try:
            # 垃圾回收
            collected = gc.collect()
            logger.info(f"Resource cleanup completed, collected {collected} objects")
        except Exception as e:
            logger.error(f"Resource cleanup failed: {e}")

# 全局资源管理器实例
resource_manager = ResourceManager()