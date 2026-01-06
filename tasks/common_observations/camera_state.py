# Copyright (c) 2025, Unitree Robotics Co., Ltd. All Rights Reserved.
# License: Apache License, Version 2.0  
"""
camera state
"""     

from __future__ import annotations

from typing import TYPE_CHECKING
import torch
import sys
import os
import threading
import queue
import time

# add the project root directory to the path, so that the shared memory tool can be imported
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from tools.shared_memory_utils import MultiImageWriter
from tools.camera_optimizer import CameraOptimizer, CameraType
from tools.clock_synchronizer import ClockSynchronizer, CameraSyncManager

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

# create the global multi-image shared memory writer
multi_image_writer = MultiImageWriter()

# create the global camera optimizer
camera_optimizer = CameraOptimizer()

# camera optimization settings
_camera_optimization_enabled = os.getenv("CAMERA_OPTIMIZATION", "1") == "1"
_task_type = os.getenv("TASK_TYPE", "pick_place")

# clock synchronization settings
_clock_sync_enabled = os.getenv("CLOCK_SYNC_ENABLED", "1") == "1"
if _clock_sync_enabled:
    _clock_synchronizer = ClockSynchronizer()
    _camera_sync_manager = CameraSyncManager(_clock_synchronizer)
else:
    _clock_synchronizer = None
    _camera_sync_manager = None

def set_writer_options(enable_jpeg: bool = False, jpeg_quality: int = 85, skip_cvtcolor: bool = False):
    try:
        multi_image_writer.set_options(enable_jpeg=enable_jpeg, jpeg_quality=jpeg_quality, skip_cvtcolor=skip_cvtcolor)
        print(f"[camera_state] writer options: jpeg={enable_jpeg}, quality={jpeg_quality}, skip_cvtcolor={skip_cvtcolor}")
    except Exception as e:
        print(f"[camera_state] failed to set writer options: {e}")


_camera_cache = {
    'available_cameras': None,
    'camera_keys': None,
    'last_scene_id': None,
    'frame_step': 0,
    'write_interval_steps': 2,
}


_return_placeholder = None
_async_queue = None
_async_thread = None
_async_started = False

def _async_writer_loop(q: "queue.Queue", writer: MultiImageWriter):
    while True:
        try:
            item = q.get()
            if item is None:
                break
            writer.write_images(item)
        except queue.Empty:
            # 队列为空时短暂休眠，避免CPU占用过高
            time.sleep(0.01)
            continue
        except Exception as e:
            print(f"[camera_state] Async writer error: {e}")
            time.sleep(0.01)  # 出错时也休眠

def _ensure_async_started():
    global _async_started, _async_queue, _async_thread
    if not _async_started:
        _async_queue = queue.Queue(maxsize=1)
        _async_thread = threading.Thread(target=_async_writer_loop, args=(_async_queue, multi_image_writer), daemon=True)
        _async_thread.start()
        _async_started = True


def get_camera_image(
    env: ManagerBasedRLEnv,
) -> dict:
    # pass
    """get multiple camera images and write them to shared memory
    
    Args:
        env: ManagerBasedRLEnv - reinforcement learning environment instance
    
    Returns:
        dict: dictionary containing multiple camera images
    """
    global _return_placeholder
    if _return_placeholder is None:
        _return_placeholder = torch.zeros((1, 480, 640, 3))


    _camera_cache['frame_step'] = (_camera_cache['frame_step'] + 1) % max(1, _camera_cache['write_interval_steps'])

    scene_id = id(env.scene)
    if _camera_cache['last_scene_id'] != scene_id:
        _camera_cache['camera_keys'] = list(env.scene.keys())
        _camera_cache['available_cameras'] = [name for name in _camera_cache['camera_keys'] if "camera" in name.lower()]
        _camera_cache['last_scene_id'] = scene_id

    # 性能优化：只在需要写入时才更新传感器
    if _camera_cache['frame_step'] == 0:
        try:
            dt = getattr(env, 'physics_dt', 0.02)
            if hasattr(env.scene, 'sensors') and env.scene.sensors:
                for sensor in env.scene.sensors.values():
                    try:
                        sensor.update(dt, force_recompute=False)
                    except Exception:
                        pass
        except Exception:
            pass
    
    # get the camera images
    images = {}
    current_timestamp = time.time()
    # env.sim.render()

    camera_keys = _camera_cache['camera_keys']
    # Head camera (front camera)
    if "front_camera" in camera_keys:
        head_image = env.scene["front_camera"].data.output["rgb"][0]  # [batch, height, width, 3]

        if head_image.device.type == 'cpu':
            head_numpy = head_image.numpy()
        else:
            head_numpy = head_image.cpu().numpy()

        images["head"] = head_numpy

        # 提交到时钟同步器
        if _camera_sync_manager:
            _camera_sync_manager.submit_camera_frame("front_camera", head_numpy, current_timestamp)

    # Left camera (left wrist camera)
    if "left_wrist_camera" in camera_keys:
        left_image = env.scene["left_wrist_camera"].data.output["rgb"][0]
        if left_image.device.type == 'cpu':
            left_numpy = left_image.numpy()
        else:
            left_numpy = left_image.cpu().numpy()

        images["left"] = left_numpy

        # 提交到时钟同步器
        if _camera_sync_manager:
            _camera_sync_manager.submit_camera_frame("left_wrist_camera", left_numpy, current_timestamp)

    # Right camera (right wrist camera)
    if "right_wrist_camera" in camera_keys:
        right_image = env.scene["right_wrist_camera"].data.output["rgb"][0]
        if right_image.device.type == 'cpu':
            right_numpy = right_image.numpy()
        else:
            right_numpy = right_image.cpu().numpy()

        images["right"] = right_numpy

        # 提交到时钟同步器
        if _camera_sync_manager:
            _camera_sync_manager.submit_camera_frame("right_wrist_camera", right_numpy, current_timestamp)
    
    # if no camera with the specified name is found, try other common camera names
    if not images:

        available_cameras = _camera_cache['available_cameras']
        if available_cameras:
            print(f"[camera_state] No standard cameras found. Available cameras: {available_cameras}")
            
            # if there are available cameras, use the first three as head, left, right
            for i, camera_name in enumerate(available_cameras[:3]):
                camera_image = env.scene[camera_name].data.output["rgb"][0]
                
               
                if camera_image.device.type == 'cpu':
                    numpy_image = camera_image.numpy()
                else:
                    numpy_image = camera_image.cpu().numpy()
                
                if i == 0:
                    images["head"] = numpy_image
                elif i == 1:
                    images["left"] = numpy_image
                elif i == 2:
                    images["right"] = numpy_image
    

    # 如果启用了时钟同步，尝试获取同步的图像
    if _camera_sync_manager and _camera_cache['frame_step'] == 0:
        try:
            synced_frames = _camera_sync_manager.wait_for_sync_frames(timeout=0.005)  # 5ms超时
            if synced_frames and len(synced_frames) == 3:  # 确保所有相机都有数据
                # 使用同步的帧数据
                images = {
                    "head": synced_frames.get("front_camera"),
                    "left": synced_frames.get("left_wrist_camera"),
                    "right": synced_frames.get("right_wrist_camera")
                }
                # 过滤None值
                images = {k: v for k, v in images.items() if v is not None}
        except Exception as e:
            print(f"[camera_state] Frame synchronization failed: {e}")

    if images and _camera_cache['frame_step'] == 0:
        # 应用相机优化（如果启用）
        if _camera_optimization_enabled:
            try:
                optimized_images = {}
                for name, image in images.items():
                    camera_type_map = {
                        "head": CameraType.FRONT,
                        "left": CameraType.WRIST_LEFT,
                        "right": CameraType.WRIST_RIGHT
                    }
                    if name in camera_type_map:
                        camera_type = camera_type_map[name]
                        optimized_params = camera_optimizer.optimize_camera_params(
                            camera_type, image, _task_type
                        )
                        # 这里可以记录优化参数，用于后续相机配置更新
                        # print(f"[camera_state] Optimized {name}: focal={optimized_params.focal_length:.1f}, "
                        #       f"exposure={optimized_params.exposure:.2f}")
                        optimized_images[name] = image  # 暂时保持原图，后续可应用优化
                    else:
                        optimized_images[name] = image
                images = optimized_images
            except Exception as e:
                print(f"[camera_state] Camera optimization failed: {e}")

        _ensure_async_started()
        try:
            if _async_queue.full():
                _async_queue.get_nowait()
            _async_queue.put_nowait(images)
        except Exception:
            pass
    elif not images:
        print("[camera_state] No camera images found in the environment")
    
    return _return_placeholder

