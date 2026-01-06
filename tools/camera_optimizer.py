# Copyright (c) 2025, Unitree Robotics Co., Ltd. All Rights Reserved.
# License: Apache License, Version 2.0
"""
相机参数动态优化器
根据场景内容和任务需求动态调整相机参数
"""

import numpy as np
import cv2
from typing import Dict, Tuple, Optional
from dataclasses import dataclass
from enum import Enum

class CameraType(Enum):
    FRONT = "front"
    WRIST_LEFT = "left_wrist"
    WRIST_RIGHT = "right_wrist"

@dataclass
class CameraMetrics:
    """相机图像质量指标"""
    brightness: float
    contrast: float
    sharpness: float
    noise_level: float
    dynamic_range: float

@dataclass
class OptimizedCameraParams:
    """优化后的相机参数"""
    focal_length: float
    f_number: float  # 光圈值
    exposure: float
    focus_distance: float
    iso: int

class CameraOptimizer:
    """相机参数动态优化器"""

    def __init__(self):
        # 不同任务的相机参数预设
        self.task_presets = {
            "pick_place": {
                CameraType.FRONT: OptimizedCameraParams(12.0, 2.8, 0.8, 1.2, 400),
                CameraType.WRIST_LEFT: OptimizedCameraParams(8.0, 2.0, 0.6, 0.3, 800),
                CameraType.WRIST_RIGHT: OptimizedCameraParams(8.0, 2.0, 0.6, 0.3, 800)
            },
            "assembly": {
                CameraType.FRONT: OptimizedCameraParams(16.0, 4.0, 1.0, 2.0, 200),
                CameraType.WRIST_LEFT: OptimizedCameraParams(12.0, 2.8, 0.8, 0.5, 400),
                CameraType.WRIST_RIGHT: OptimizedCameraParams(12.0, 2.8, 0.8, 0.5, 400)
            },
            "inspection": {
                CameraType.FRONT: OptimizedCameraParams(20.0, 5.6, 1.2, 3.0, 100),
                CameraType.WRIST_LEFT: OptimizedCameraParams(16.0, 4.0, 1.0, 0.8, 200),
                CameraType.WRIST_RIGHT: OptimizedCameraParams(16.0, 4.0, 1.0, 0.8, 200)
            }
        }

        # 自适应调整历史
        self.adjustment_history = {}

    def analyze_image_quality(self, image: np.ndarray) -> CameraMetrics:
        """分析图像质量指标"""
        if image is None or image.size == 0:
            return CameraMetrics(0, 0, 0, 1.0, 0)

        # 转换为灰度图进行分析
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image

        # 计算亮度
        brightness = np.mean(gray) / 255.0

        # 计算对比度
        contrast = np.std(gray) / 128.0

        # 计算锐度 (使用Laplacian方差)
        sharpness = cv2.Laplacian(gray, cv2.CV_64F).var()

        # 估算噪声水平 (使用高通滤波器)
        kernel = np.array([[-1,-1,-1], [-1,8,-1], [-1,-1,-1]])
        noise_map = cv2.filter2D(gray, -1, kernel)
        noise_level = np.mean(np.abs(noise_map)) / 255.0

        # 计算动态范围
        hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
        hist = hist.flatten()
        cumulative = np.cumsum(hist)
        low_percentile = np.where(cumulative > len(gray.flatten()) * 0.05)[0]
        high_percentile = np.where(cumulative > len(gray.flatten()) * 0.95)[0]
        if len(low_percentile) > 0 and len(high_percentile) > 0:
            dynamic_range = (high_percentile[0] - low_percentile[0]) / 255.0
        else:
            dynamic_range = 0.5

        return CameraMetrics(
            brightness=float(brightness),
            contrast=float(contrast),
            sharpness=float(sharpness),
            noise_level=float(noise_level),
            dynamic_range=float(dynamic_range)
        )

    def optimize_camera_params(self, camera_type: CameraType, image: np.ndarray,
                             task_type: str = "pick_place") -> OptimizedCameraParams:
        """根据图像质量动态优化相机参数"""

        # 获取基础预设
        base_params = self.task_presets.get(task_type, self.task_presets["pick_place"])[camera_type]

        # 分析当前图像质量
        metrics = self.analyze_image_quality(image)

        # 基于质量指标调整参数
        optimized = OptimizedCameraParams(
            focal_length=base_params.focal_length,
            f_number=base_params.f_number,
            exposure=base_params.exposure,
            focus_distance=base_params.focus_distance,
            iso=base_params.iso
        )

        # 亮度调整
        if metrics.brightness < 0.3:
            # 太暗，增加曝光和ISO
            optimized.exposure = min(base_params.exposure * 1.5, 2.0)
            optimized.iso = min(base_params.iso * 2, 1600)
        elif metrics.brightness > 0.7:
            # 太亮，减少曝光
            optimized.exposure = max(base_params.exposure * 0.7, 0.1)
            optimized.iso = max(base_params.iso // 2, 100)

        # 对比度调整
        if metrics.contrast < 0.3:
            # 对比度低，稍微增加光圈
            optimized.f_number = max(base_params.f_number * 0.8, 1.4)

        # 锐度调整
        if metrics.sharpness < 100:
            # 图像模糊，增加焦距或调整对焦
            optimized.focal_length = min(base_params.focal_length * 1.2, 35.0)
            optimized.focus_distance = max(base_params.focus_distance * 0.8, 0.1)

        # 噪声水平调整
        if metrics.noise_level > 0.1:
            # 噪声大，降低ISO
            optimized.iso = max(base_params.iso // 2, 100)

        # 记录调整历史
        key = f"{camera_type.value}_{task_type}"
        if key not in self.adjustment_history:
            self.adjustment_history[key] = []
        self.adjustment_history[key].append((metrics, optimized))
        if len(self.adjustment_history[key]) > 10:  # 保持最近10次记录
            self.adjustment_history[key].pop(0)

        return optimized

    def get_camera_config_from_params(self, params: OptimizedCameraParams) -> Dict:
        """将优化参数转换为相机配置"""
        return {
            "focal_length": params.focal_length,
            "horizontal_aperture": params.focal_length / params.f_number,  # 计算水平光圈
            "exposure": params.exposure,
            "focus_distance": params.focus_distance,
            # ISO影响曝光，但这里主要用于参考
            "iso": params.iso
        }
