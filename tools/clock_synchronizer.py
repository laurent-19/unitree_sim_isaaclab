# Copyright (c) 2025, Unitree Robotics Co., Ltd. All Rights Reserved.
# License: Apache License, Version 2.0
"""
时钟同步管理器
确保多相机、渲染和控制循环的时间同步和数据一致性
"""

import time
import threading
from typing import Dict, List, Optional, Callable
from dataclasses import dataclass
from collections import deque
import numpy as np

@dataclass
class TimestampedData:
    """带时间戳的数据"""
    timestamp: float
    data: any
    sequence_id: int

@dataclass
class SynchronizationConfig:
    """同步配置"""
    max_time_window: float = 0.033  # 30ms时间窗口
    max_sequence_diff: int = 2      # 最大序列号差异
    interpolation_enabled: bool = True
    extrapolation_enabled: bool = False

class ClockSynchronizer:
    """时钟同步管理器"""

    def __init__(self, config: SynchronizationConfig = None):
        self.config = config or SynchronizationConfig()
        self.master_clock = time.perf_counter

        # 数据缓冲区 - 为每个数据流维护缓冲区
        self.data_buffers: Dict[str, deque] = {}
        self.buffer_lock = threading.Lock()

        # 同步统计
        self.sync_stats = {
            "total_sync_operations": 0,
            "successful_syncs": 0,
            "failed_syncs": 0,
            "avg_sync_latency": 0.0
        }

        # 回调函数
        self.sync_callbacks: List[Callable] = []

        print("[ClockSynchronizer] Initialized with max_time_window=%.3fs",
              self.config.max_time_window)

    def register_data_stream(self, stream_name: str, buffer_size: int = 10):
        """注册数据流"""
        with self.buffer_lock:
            self.data_buffers[stream_name] = deque(maxlen=buffer_size)
        print(f"[ClockSynchronizer] Registered data stream: {stream_name}")

    def submit_data(self, stream_name: str, data: any, timestamp: float = None,
                   sequence_id: Optional[int] = None) -> bool:
        """提交数据到指定流"""
        if stream_name not in self.data_buffers:
            print(f"[ClockSynchronizer] Unknown stream: {stream_name}")
            return False

        if timestamp is None:
            timestamp = self.master_clock()

        if sequence_id is None:
            # 生成序列号
            with self.buffer_lock:
                buffer = self.data_buffers[stream_name]
                if buffer:
                    sequence_id = buffer[-1].sequence_id + 1
                else:
                    sequence_id = 0

        timestamped_data = TimestampedData(timestamp, data, sequence_id)

        with self.buffer_lock:
            self.data_buffers[stream_name].append(timestamped_data)

        return True

    def get_synchronized_data(self, target_timestamp: float = None,
                            required_streams: List[str] = None) -> Dict[str, TimestampedData]:
        """获取同步后的数据"""
        if target_timestamp is None:
            target_timestamp = self.master_clock()

        if required_streams is None:
            required_streams = list(self.data_buffers.keys())

        synchronized_data = {}
        sync_start_time = self.master_clock()

        with self.buffer_lock:
            for stream_name in required_streams:
                if stream_name not in self.data_buffers:
                    continue

                buffer = self.data_buffers[stream_name]
                if not buffer:
                    continue

                # 寻找最接近目标时间戳的数据
                best_match = None
                min_time_diff = float('inf')

                for item in buffer:
                    time_diff = abs(item.timestamp - target_timestamp)
                    if time_diff < min_time_diff:
                        min_time_diff = time_diff
                        best_match = item

                # 检查是否在允许的时间窗口内
                if best_match and min_time_diff <= self.config.max_time_window:
                    synchronized_data[stream_name] = best_match

        # 更新同步统计
        self.sync_stats["total_sync_operations"] += 1
        sync_latency = self.master_clock() - sync_start_time
        self.sync_stats["avg_sync_latency"] = (
            self.sync_stats["avg_sync_latency"] * (self.sync_stats["total_sync_operations"] - 1) +
            sync_latency
        ) / self.sync_stats["total_sync_operations"]

        if len(synchronized_data) == len(required_streams):
            self.sync_stats["successful_syncs"] += 1
            # 触发同步成功回调
            for callback in self.sync_callbacks:
                try:
                    callback(synchronized_data)
                except Exception as e:
                    print(f"[ClockSynchronizer] Sync callback error: {e}")
        else:
            self.sync_stats["failed_syncs"] += 1

        return synchronized_data

    def interpolate_data(self, stream_name: str, target_timestamp: float) -> Optional[TimestampedData]:
        """在数据流中插值获取目标时间戳的数据"""
        if not self.config.interpolation_enabled:
            return None

        with self.buffer_lock:
            buffer = self.data_buffers.get(stream_name)
            if not buffer or len(buffer) < 2:
                return None

            # 找到目标时间戳前后的两个数据点
            before_point = None
            after_point = None

            for item in buffer:
                if item.timestamp <= target_timestamp:
                    before_point = item
                elif item.timestamp > target_timestamp:
                    after_point = item
                    break

            if before_point is None or after_point is None:
                return before_point or after_point

            # 线性插值
            time_diff = after_point.timestamp - before_point.timestamp
            if time_diff == 0:
                return before_point

            ratio = (target_timestamp - before_point.timestamp) / time_diff

            # 对于数值数据进行插值
            if isinstance(before_point.data, (int, float)) and isinstance(after_point.data, (int, float)):
                interpolated_value = before_point.data + ratio * (after_point.data - before_point.data)
            elif isinstance(before_point.data, np.ndarray) and isinstance(after_point.data, np.ndarray):
                # 对于数组数据进行插值
                interpolated_value = before_point.data + ratio * (after_point.data - before_point.data)
            else:
                # 对于非数值数据，返回最接近的点
                return before_point if ratio < 0.5 else after_point

            return TimestampedData(
                timestamp=target_timestamp,
                data=interpolated_value,
                sequence_id=before_point.sequence_id  # 使用前一个序列号
            )

    def get_sync_stats(self) -> Dict:
        """获取同步统计信息"""
        total_ops = self.sync_stats["total_sync_operations"]
        if total_ops == 0:
            success_rate = 0.0
        else:
            success_rate = self.sync_stats["successful_syncs"] / total_ops

        return {
            **self.sync_stats,
            "success_rate": success_rate,
            "buffer_sizes": {name: len(buffer) for name, buffer in self.data_buffers.items()}
        }

    def add_sync_callback(self, callback: Callable):
        """添加同步成功回调"""
        self.sync_callbacks.append(callback)

    def clear_old_data(self, max_age: float = 1.0):
        """清除过旧的数据"""
        current_time = self.master_clock()
        cutoff_time = current_time - max_age

        with self.buffer_lock:
            for stream_name, buffer in self.data_buffers.items():
                # 从头部移除过旧的数据
                while buffer and buffer[0].timestamp < cutoff_time:
                    buffer.popleft()

class CameraSyncManager:
    """相机同步管理器"""

    def __init__(self, synchronizer: ClockSynchronizer):
        self.synchronizer = synchronizer
        self.camera_streams = ["front_camera", "left_wrist_camera", "right_wrist_camera"]

        # 注册相机数据流
        for stream in self.camera_streams:
            self.synchronizer.register_data_stream(stream, buffer_size=5)

    def submit_camera_frame(self, camera_name: str, frame: np.ndarray,
                           timestamp: float = None) -> bool:
        """提交相机帧数据"""
        return self.synchronizer.submit_data(camera_name, frame, timestamp)

    def get_synchronized_frames(self, target_timestamp: float = None) -> Dict[str, np.ndarray]:
        """获取同步的相机帧"""
        sync_data = self.synchronizer.get_synchronized_data(
            target_timestamp, self.camera_streams
        )

        return {name: data.data for name, data in sync_data.items()}

    def wait_for_sync_frames(self, timeout: float = 0.1) -> Optional[Dict[str, np.ndarray]]:
        """等待所有相机帧同步完成"""
        start_time = time.time()
        while time.time() - start_time < timeout:
            frames = self.get_synchronized_frames()
            if len(frames) == len(self.camera_streams):
                return frames
            time.sleep(0.001)  # 短暂等待
        return None
