
# Copyright (c) 2025, Unitree Robotics Co., Ltd. All Rights Reserved.
# License: Apache License, Version 2.0  
#!/usr/bin/env python3
# main.py
import os

project_root = os.path.dirname(os.path.abspath(__file__))
os.environ["PROJECT_ROOT"] = project_root

import argparse
import contextlib
import time
import sys
import signal
import torch
import gymnasium as gym
from pathlib import Path

# Isaac Lab AppLauncher
from isaaclab.app import AppLauncher

from teleimager.image_server import run_isaacsim_server
from dds.dds_create import create_dds_objects,create_dds_objects_replay
# add command line arguments
parser = argparse.ArgumentParser(description="Unitree Simulation")
parser.add_argument("--task", type=str, default="Isaac-PickPlace-G129-Head-Waist-Fix", help="task name")
parser.add_argument("--action_source", type=str, default="dds", 
                   choices=["dds", "file", "trajectory", "policy", "replay","dds_wholebody"], 
                   help="Action source")


parser.add_argument("--robot_type", type=str, default="g129", help="robot type")
parser.add_argument("--enable_dex1_dds", action="store_true", help="enable gripper DDS")
parser.add_argument("--enable_dex3_dds", action="store_true", help="enable dexterous hand DDS")
parser.add_argument("--enable_inspire_dds", action="store_true", help="enable inspire hand DDS")
parser.add_argument("--stats_interval", type=float, default=10.0, help="statistics print interval (seconds)")

parser.add_argument("--file_path", type=str, default="/home/unitree/Code/xr_teleoperate/teleop/utils/data", help="file path (when action_source=file)")
parser.add_argument("--generate_data_dir", type=str, default="./data", help="save data dir")
parser.add_argument("--generate_data", action="store_true", default=False, help="generate data")
parser.add_argument("--rerun_log", action="store_true", default=False, help="rerun log")
parser.add_argument("--replay_data",  action="store_true", default=False, help="replay data")

parser.add_argument("--modify_light",  action="store_true", default=False, help="modify light")
parser.add_argument("--modify_camera",  action="store_true", default=False,    help="modify camera")

# performance analysis parameters
parser.add_argument("--step_hz", type=int, default=100, help="control frequency")
parser.add_argument("--enable_profiling", action="store_true", default=True, help="enable performance analysis")
parser.add_argument("--profile_interval", type=int, default=500, help="performance analysis report interval (steps)")

parser.add_argument("--model_path", type=str, default="assets/model/policy.onnx", help="model path")
parser.add_argument("--reward_interval", type=int, default=10, help="step interval for reward calculation")
parser.add_argument("--enable_wholebody_dds", action="store_true", default=False, help="enable wh dds")

parser.add_argument("--physics_dt", type=float, default=None, help="physics time step, e.g., 0.005")
parser.add_argument("--render_interval", type=int, default=None, help="render interval steps (>=1)")
parser.add_argument("--camera_write_interval", type=int, default=None, help="camera write interval steps (>=1)")


parser.add_argument(
    "--no_render",
    action="store_true",
    default=False,
    help="disable rendering updates entirely (overrides render interval)",
)
parser.add_argument("--solver_iterations", type=int, default=None, help="physx solver iteration count (e.g., 4)")
parser.add_argument("--gravity_z", type=float, default=None, help="override gravity z (e.g., -9.8)")
parser.add_argument("--skip_cvtcolor", action="store_true", default=False, help="skip cv2.cvtColor if upstream already BGR")

parser.add_argument("--camera_jpeg", action="store_true", default=True, help="enable JPEG compression for camera frames")
parser.add_argument("--camera_jpeg_quality", type=int, default=85, help="JPEG quality (1-100)")

parser.add_argument("--physx_substeps", type=int, default=None, help="physx substeps per step")
parser.add_argument("--camera_include", type=str, default="front_camera,left_wrist_camera,right_wrist_camera", help="comma-separated camera names to enable")
parser.add_argument("--camera_exclude", type=str, default="world_camera", help="comma-separated camera names to disable")

parser.add_argument("--env_reward_interval", type=int, default=5, help="environment reward compute interval (steps)")
parser.add_argument("--seed", type=int, default=42, help="environment seed")

# add AppLauncher parameters
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()


if args_cli.enable_dex3_dds and args_cli.enable_dex1_dds and args_cli.enable_inspire_dds:
    logger.error("Cannot enable dex3_dds, dex1_dds and inspire_dds at the same time")
    logger.error("Please select one of the gripper options")
    sys.exit(1)


import pinocchio                 
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

from layeredcontrol.robot_control_system import (
    RobotController, 
    ControlConfig,
)

from dds.reset_pose_dds import *
import tasks
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg

from tools.augmentation_utils import (
    update_light,
    batch_augment_cameras_by_name,
)

from tools.data_json_load import sim_state_to_json
from dds.sim_state_dds import *
from action_provider.create_action_provider import create_action_provider
from tools.get_stiffness import get_robot_stiffness_from_env
from tools.get_reward import get_step_reward_value,get_current_rewards
from tools.render_optimizer import RenderOptimizer, TaskType as RenderTaskType
from tools.system_initializer import init_system, cleanup_system
from tools.config_manager import config_manager, TaskType as ConfigTaskType, RobotType
from tools.logger_manager import logger_manager, get_logger, LogCategory, LogLevel
from tools.performance_monitor import performance_monitor
from tools.resource_manager import resource_manager

def setup_signal_handlers(controller,dds_manager=None,image_server=None):
    """set signal handlers"""
    def signal_handler(signum, frame):
        logger_manager.log(LogLevel.WARNING, f"Received signal {signum}, stopping controller",
                          LogCategory.SYSTEM, "signal_handler", metadata={'signal': signum})
        try:
            controller.stop()
            logger_manager.log(LogLevel.INFO, "Controller stopped successfully",
                              LogCategory.SYSTEM, "signal_handler")
        except Exception as e:
            logger_manager.log(LogLevel.ERROR, f"Failed to stop controller: {e}",
                              LogCategory.SYSTEM, "signal_handler", metadata={'error_type': type(e).__name__})
        try:
            if dds_manager is not None:
                dds_manager.stop_all_communication()
                logger_manager.log(LogLevel.INFO, "DDS communication stopped successfully",
                                  LogCategory.SYSTEM, "signal_handler")
        except Exception as e:
            logger_manager.log(LogLevel.ERROR, f"Failed to stop DDS: {e}",
                              LogCategory.SYSTEM, "signal_handler", metadata={'error_type': type(e).__name__})
        try:
            if image_server is not None:
                image_server.stop()
                logger_manager.log(LogLevel.INFO, "Image server stopped successfully",
                                  LogCategory.SYSTEM, "signal_handler")
        except Exception as e:
            logger_manager.log(LogLevel.ERROR, f"Failed to stop image server: {e}",
                              LogCategory.SYSTEM, "signal_handler", metadata={'error_type': type(e).__name__})
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)



def main():
    """main function"""
    # 初始化日志系统
    logger = get_logger("main")

    # 映射任务类型和机器人类型
    task_type_map = {
        "replay": ConfigTaskType.REPLAY,
        "file": ConfigTaskType.EVALUATION,
        "trajectory": ConfigTaskType.EVALUATION,
        "policy": ConfigTaskType.EVALUATION,
        "dds": ConfigTaskType.REAL_TIME,
        "dds_wholebody": ConfigTaskType.REAL_TIME
    }
    robot_type_map = {
        "g129": RobotType.G129,
        "h1_2": RobotType.H1_2
    }

    current_task_type = task_type_map.get(args_cli.action_source, ConfigTaskType.DEFAULT)
    current_robot_type = robot_type_map.get(args_cli.robot_type, RobotType.G129)

    # 准备配置覆盖
    config_overrides = {
        "system": {
            "task_type": current_task_type,
            "robot_type": current_robot_type
        },
        "performance": {
            "step_hz": args_cli.step_hz,
            "profiling_enabled": args_cli.enable_profiling
        },
        "camera": {
            "camera_jpeg": args_cli.camera_jpeg,
            "camera_jpeg_quality": args_cli.camera_jpeg_quality
        }
    }

    # 初始化整个系统
    if not init_system(config_overrides):
        logger.error("System initialization failed, exiting")
        return

    logger.info(f"Configuration initialized: task_type={current_task_type.value}, "
               f"robot_type={current_robot_type.value}")

    # 启动性能监控
    performance_monitor.start_monitoring()
    benchmark_id = performance_monitor.start_benchmark(f"simulation_{current_task_type.value}")
    logger.info(f"Performance monitoring started, benchmark: {benchmark_id}")

    # import cProfile
    # import pstats
    # import io
    # profiler = cProfile.Profile()
    # profiler.enable()
    import os
    import atexit
    try:
        os.setpgrp()
        current_pgid = os.getpgrp()
        logger.debug(f"Setting process group: {current_pgid}")

        def cleanup_process_group():
            try:
                logger.info(f"Cleaning up process group: {current_pgid}")
                import signal
                os.killpg(current_pgid, signal.SIGTERM)
                logger.debug("Process group cleanup signal sent")
            except Exception as e:
                logger.error(f"Failed to clean up process group: {e}", extra={'pgid': current_pgid})

        atexit.register(cleanup_process_group)

    except Exception as e:
        logger.warning(f"Failed to set process group: {e}", extra={'error_type': type(e).__name__})
    logger.info("=" * 60)
    logger.info("Robot control system started")
    logger.info(f"Task: {args_cli.task}")
    logger.info(f"Action source: {args_cli.action_source}")
    logger.info("=" * 60)

    # 初始化智能渲染优化器（转换任务类型）
    from tools.render_optimizer import RenderOptimizer, TaskType as RenderTaskType
    render_task_type_map = {
        ConfigTaskType.EVALUATION: RenderTaskType.EVALUATION,
        ConfigTaskType.REAL_TIME: RenderTaskType.REAL_TIME,
        ConfigTaskType.REPLAY: RenderTaskType.REPLAY
    }
    render_task_type = render_task_type_map.get(current_task_type, RenderTaskType.REAL_TIME)
    render_optimizer = RenderOptimizer(render_task_type)
    logger.info(f"Initialized render optimizer for task type: {render_task_type.value}")

    # 启动性能基准测试
    benchmark_id = performance_monitor.start_benchmark(f"simulation_{current_task_type.value}")
    logger.info(f"Performance benchmark started: {benchmark_id}")

    # parse environment configuration
    logger.info("Starting environment configuration parsing")
    try:
        env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=1)
        env_cfg.env_name = args_cli.task
        logger.info(f"Environment configuration parsed successfully for task: {args_cli.task}")
    except Exception as e:
        logger.error(f"Failed to parse environment configuration: {e}", extra={
            'task': args_cli.task,
            'device': args_cli.device,
            'error_type': type(e).__name__
        })
        return

    # create environment
    logger.info("Starting environment creation")
    try:
        logger.debug(f"Setting environment seed to: {args_cli.seed}")
        env_cfg.seed = args_cli.seed
        logger.info(f"Creating gym environment: {args_cli.task}")
        env = gym.make(args_cli.task, cfg=env_cfg).unwrapped
        env.seed(args_cli.seed)
        logger.info("Environment created successfully")
        try:
            sensors_dict = getattr(env.scene, "sensors", {})
            if sensors_dict:
                logger.info("Sensors in the environment:")
                for name, sensor in sensors_dict.items():
                    logger.debug(f"Sensor: {name} = {sensor}")
                logger.info("="*60)
        except Exception as e:
            logger.warning(f"Failed to list sensors: {e}", extra={'error_type': type(e).__name__})
        logger.info("Environment created successfully")
        try:
            env._reward_interval = max(1, int(args_cli.env_reward_interval))
            env._reward_counter = 0
            env._reward_last = None
            logger.debug(f"Environment reward compute interval set to {env._reward_interval} steps")
        except Exception as e:
            logger.warning(f"Failed to set reward interval: {e}", extra={'error_type': type(e).__name__})
        if args_cli.physics_dt is not None:
            try:
                env.sim.set_substep_time(args_cli.physics_dt)
                logger.debug(f"Physics dt set to {args_cli.physics_dt}")
            except Exception:
                try:
                    env.sim.dt = args_cli.physics_dt
                    logger.debug(f"Physics dt assigned to env.sim.dt={args_cli.physics_dt}")
                except Exception as e:
                    logger.warning(f"Failed to set physics dt: {e}", extra={'error_type': type(e).__name__})

        # 获取智能渲染配置
        optimal_config = render_optimizer.get_optimal_config()
        logger.info(f"Using optimized render config: interval={optimal_config.render_interval}, "
                   f"camera_period={optimal_config.camera_update_period:.3f}s, "
                   f"dds_rate={optimal_config.dds_publish_rate}Hz, "
                   f"jpeg={optimal_config.enable_jpeg}(q={optimal_config.jpeg_quality})")

        headless_mode = bool(getattr(args_cli, "headless", False))
        render_interval = None
        if args_cli.render_interval is not None:
            # 使用命令行参数或智能配置中的较大值
            render_interval = max(optimal_config.render_interval, int(args_cli.render_interval))
        else:
            render_interval = optimal_config.render_interval
        try:
            if args_cli.no_render:
                env.sim.render_interval = 1_000_000
                env.sim.render_mode = "offscreen"
                logger.info("Rendering disabled via --no_render")
            elif headless_mode:
                env.sim.render_mode = "offscreen"
                env.sim.render_interval = render_interval or 1
                logger.debug(f"Headless offscreen rendering every {env.sim.render_interval} steps")
            elif render_interval is not None:
                env.sim.render_interval = render_interval
                logger.debug(f"Render interval set to {env.sim.render_interval}")
        except Exception as e:
            logger.warning(f"Failed to configure rendering: {e}", extra={'error_type': type(e).__name__})
        if args_cli.camera_write_interval is not None:
            try:
                import tasks.common_observations.camera_state as cam_state
                cam_state._camera_cache['write_interval_steps'] = max(1, int(args_cli.camera_write_interval))
                logger.debug(f"Camera write interval steps set to {cam_state._camera_cache['write_interval_steps']}")
            except Exception as e:
                logger.warning(f"Failed to set camera write interval: {e}", extra={'error_type': type(e).__name__})

        try:
            if args_cli.solver_iterations is not None:
                env.sim.physx.solver_iteration_count = int(args_cli.solver_iterations)
                logger.debug(f"Solver iteration count set to {env.sim.physx.solver_iteration_count}")
            if args_cli.physx_substeps is not None:
                try:
                    env.sim.physx.substeps = int(args_cli.physx_substeps)
                except Exception:
                    try:
                        env.sim.set_substeps(int(args_cli.physx_substeps))
                    except Exception:
                        pass
                logger.debug(f"PhysX substeps set to {args_cli.physx_substeps}")
            if args_cli.gravity_z is not None:
                g = float(args_cli.gravity_z)
                env.sim.physx.gravity = (0.0, 0.0, g)
                logger.debug(f"Gravity set to {env.sim.physx.gravity}")
        except Exception as e:
            logger.warning(f"Failed to set PhysX parameters: {e}", extra={'error_type': type(e).__name__})
        if args_cli.skip_cvtcolor:
            os.environ["CAMERA_SKIP_CVTCOLOR"] = "1"

        # 设置相机优化环境变量
        os.environ["CAMERA_UPDATE_PERIOD"] = str(optimal_config.camera_update_period)
        os.environ["TASK_TYPE"] = current_task_type.value
        os.environ["CAMERA_OPTIMIZATION"] = "1" if config_manager.get_config_value("render", "camera_optimization") else "0"
        os.environ["CLOCK_SYNC_ENABLED"] = "1" if config_manager.get_config_value("render", "clock_sync_enabled") else "0"

        try:
            import tasks.common_observations.camera_state as cam_state
            # 使用智能配置或命令行参数
            enable_jpeg = optimal_config.enable_jpeg if not args_cli.camera_jpeg else bool(args_cli.camera_jpeg)
            jpeg_quality = optimal_config.jpeg_quality if not args_cli.camera_jpeg else int(args_cli.camera_jpeg_quality)
            cam_state.set_writer_options(enable_jpeg=enable_jpeg, jpeg_quality=jpeg_quality, skip_cvtcolor=args_cli.skip_cvtcolor)
            include = [n.strip() for n in (args_cli.camera_include or "").split(',') if n.strip()]
            exclude = [n.strip() for n in (args_cli.camera_exclude or "").split(',') if n.strip()]
            try:
                cam_state.set_camera_allowlist(include)
            except Exception:
                pass
            try:
                sensors_dict = getattr(env.scene, "sensors", {})
                for name, sensor in sensors_dict.items():
                    lname = name.lower()
                    if "camera" not in lname:
                        continue
                    if exclude and name in exclude:
                        for attr_name, value in [("enabled", False), ("is_enabled", False)]:
                            if hasattr(sensor, attr_name):
                                try:
                                    setattr(sensor, attr_name, value)
                                except Exception:
                                    pass
                        for meth in ("set_active", "disable", "pause"):
                            if hasattr(sensor, meth):
                                try:
                                    getattr(sensor, meth)(False)
                                except Exception:
                                    pass
                        for attr_name in ("update_period", "_update_period"):
                            if hasattr(sensor, attr_name):
                                try:
                                    setattr(sensor, attr_name, 1e6)
                                except Exception:
                                    pass
                    elif include and name not in include:
                        for attr_name in ("update_period", "_update_period"):
                            if hasattr(sensor, attr_name):
                                try:
                                    setattr(sensor, attr_name, 1e6)
                                except Exception:
                                    pass
            except Exception as e:
                logger.warning(f"Failed to tune camera sensors: {e}", extra={'error_type': type(e).__name__})
        except Exception as e:
            logger.warning(f"Failed to apply camera writer options: {e}", extra={'error_type': type(e).__name__})
    except Exception as e:
        logger.error(f"Failed to create environment: {e}", extra={
            'task': args_cli.task,
            'device': args_cli.device,
            'seed': args_cli.seed,
            'error_type': type(e).__name__,
            'error_location': 'environment_creation'
        })
        return
    
    # get robot stiffness and damping parameters from runtime environment
    logger.info("="*60)
    logger.info("🔍 Getting robot stiffness and damping parameters from runtime environment")
    logger.info("="*60)
    
    try:
        stiffness_data = get_robot_stiffness_from_env(env)
        if stiffness_data:
            logger.info("✅ Successfully got robot parameters!")
        else:
            logger.warning("⚠️ Failed to get robot parameters, will try again after environment reset")
    except Exception as e:
        logger.warning(f"⚠️ Error getting robot parameters: {e}", extra={'error_type': type(e).__name__})

    logger.info("="*60)
    
    if not getattr(args_cli, "headless", False) and not args_cli.no_render:
        logger.info("")
        logger.info("***  Please left-click on the Sim window to activate rendering. ***")
        logger.info("")
    else:
        logger.info("")
        logger.info("***  Running without GUI; rendering handled offscreen. ***")
        logger.info("")
    # reset environment
    if args_cli.modify_light:
        update_light(
            prim_path="/World/light",
            color=(0.75, 0.75, 0.75),
            intensity=500.0,
            # position=(1.0, 2.0, 3.0),
            radius=0.1,
            enabled=True,
            cast_shadows=True
        )
    if args_cli.modify_camera:
        batch_augment_cameras_by_name(
            names=["front_cam"],
            focal_length=3.0,
            horizontal_aperture=22.0,
            vertical_aperture=16.0,
            exposure=0.8,                
            focus_distance=1.2
        )
    env.sim.reset()
    env.reset()
    
    # create simplified control configuration
    try:    
        control_config = ControlConfig(
            step_hz=args_cli.step_hz,
            replay_mode=args_cli.replay_data
        )
    except Exception as e:
        print(f"Failed to create control configuration: {e}")
        return
    
    # create controller

    if not args_cli.replay_data:
        logger.info("Creating image server")
        try:
            image_server = run_isaacsim_server()
            logger.info("Image server created successfully")
        except Exception as e:
            logger.error(f"Failed to create image server: {e}", extra={
                'error_type': type(e).__name__,
                'replay_mode': args_cli.replay_data
            })
            return

        logger.info("Creating DDS objects")
        try:
            reset_pose_dds,sim_state_dds,dds_manager = create_dds_objects(args_cli,env)
            logger.info("DDS objects created successfully")
        except Exception as e:
            logger.error(f"Failed to create DDS objects: {e}", extra={
                'error_type': type(e).__name__,
                'robot_type': args_cli.robot_type,
                'enable_dex1_dds': args_cli.enable_dex1_dds,
                'enable_dex3_dds': args_cli.enable_dex3_dds,
                'enable_inspire_dds': args_cli.enable_inspire_dds
            })
            return
    else:
        logger.info("Creating DDS objects for replay mode")
        try:
            create_dds_objects_replay(args_cli,env)
            logger.info("DDS objects for replay created successfully")
        except Exception as e:
            logger.error(f"Failed to create DDS objects for replay: {e}", extra={
                'error_type': type(e).__name__,
                'replay_mode': True
            })
            return
        from tools.data_json_load import get_data_json_list
        print("========= get data json list =========")
        data_idx=0
        data_json_list = get_data_json_list(args_cli.file_path)
        if args_cli.action_source != "replay":
            args_cli.action_source = "replay"
        print("========= get data json list success =========")
    # create action provider
    logger.info(f"Creating action provider: {args_cli.action_source}")
    try:
        logger.debug(f"Task: {args_cli.task}, Replay mode: {args_cli.replay_data}")
        if not args_cli.replay_data and ("Wholebody" in args_cli.task or args_cli.enable_wholebody_dds):
            logger.info("Detected wholebody task, switching to dds_wholebody mode")
            args_cli.action_source = "dds_wholebody"
            args_cli.enable_wholebody_dds = True
            control_config.use_rl_action_mode = True

        action_provider = create_action_provider(env, args_cli)
        if action_provider is None:
            logger.error("Action provider creation returned None", extra={
                'action_source': args_cli.action_source,
                'task': args_cli.task
            })
            return
        logger.info(f"Action provider created successfully: {type(action_provider).__name__}")
    except Exception as e:
        logger.error(f"Failed to create action provider: {e}", extra={
            'action_source': args_cli.action_source,
            'task': args_cli.task,
            'error_type': type(e).__name__,
            'replay_mode': args_cli.replay_data
        })
        return
    
    # set action provider
    logger.info("========= create controller =========")
    controller = RobotController(env, control_config)
    controller.set_action_provider(action_provider)
    logger.info("========= create controller success =========")

    # configure performance analysis
    if args_cli.enable_profiling:
        controller.set_profiling(True, args_cli.profile_interval)
        logger.debug(f"performance analysis enabled, report every {args_cli.profile_interval} steps")
    else:
        controller.set_profiling(False)
        logger.debug("performance analysis disabled")


    # Signal handlers are now registered by system initializer
        
    logger.info("Note: The DDS in Sim transmits messages on channel 1. Please ensure that other DDS instances use the same channel for message exchange by setting: ChannelFactoryInitialize(1).")
    try:
        # start controller - start asynchronous components
        logger.info("========= start controller =========")
        controller.start()
        logger.info("========= start controller success =========")
        
        # main loop - execute in main thread to support rendering
        last_stats_time = time.time()
        loop_start_time = time.time()
        loop_count = 0
        last_loop_time = time.time()
        recent_loop_times = []  # for calculating moving average frequency
        
        
        reward_interval = max(1, args_cli.reward_interval)

        # 性能优化：减少DDS状态更新的频率
        state_update_interval = max(1, int(100 / optimal_config.dds_publish_rate))  # 根据DDS频率调整状态更新间隔

        # use torch.inference_mode() and exception suppression
        from tools.system_initializer import is_shutdown_requested
        with contextlib.suppress(KeyboardInterrupt), torch.inference_mode():
            while simulation_app.is_running() and controller.is_running and not is_shutdown_requested():
                current_time = time.time()
                loop_count += 1
                if not args_cli.replay_data:
                    # 性能优化：只在特定间隔更新状态，避免过于频繁的DDS通信
                    if loop_count % state_update_interval == 0:
                        try:
                            env_state = env.scene.get_state()
                            env_state_json =  sim_state_to_json(env_state)
                            sim_state = {"init_state":env_state_json,"task_name":args_cli.task}
                        except Exception as e:
                            logger.error(f"Failed to get env state: {e}", extra={'error_type': type(e).__name__})
                            raise e
                        try:
                        # sim_state = json.dumps(sim_state)
                            sim_state_dds.write_sim_state_data(sim_state)
                        except Exception as e:
                            logger.error(f"Failed to write sim state: {e}", extra={'error_type': type(e).__name__})
                            raise e

                    # 命令获取仍然保持高频，以确保实时响应
                    try:
                        reset_pose_cmd = reset_pose_dds.get_reset_pose_command()
                    except Exception as e:
                        logger.error(f"Failed to get reset pose command: {e}", extra={'error_type': type(e).__name__})
                        raise e
                    # Compute current reward values manually if needed for debugging
                    try:
                        if (loop_count % reward_interval) == 0:
                            pass
                            # current_reward = get_step_reward_value(env)
                    except Exception as e:
                        logger.warning(f"Reward calculation failed: {e}", extra={'error_type': type(e).__name__})

                    if reset_pose_cmd is not None:
                        try:
                            reset_category = reset_pose_cmd.get("reset_category")
                            if (args_cli.enable_wholebody_dds and (reset_category == '1' or reset_category == '2')) or (not args_cli.enable_wholebody_dds and reset_category == '1'):
                                logger.info("Resetting object")
                                env_cfg.event_manager.trigger("reset_object_self", env)
                                reset_pose_dds.write_reset_pose_command(-1)
                            elif reset_category == '2' and not args_cli.enable_wholebody_dds:
                                logger.info("Resetting all")
                                env_cfg.event_manager.trigger("reset_all_self", env)
                                reset_pose_dds.write_reset_pose_command(-1)
                        except Exception as e:
                            logger.error(f"Failed to write reset pose command: {e}", extra={'error_type': type(e).__name__})
                            raise e
                else:
                    if action_provider.get_start_loop() and data_idx<len(data_json_list):
                        logger.debug(f"Processing data index: {data_idx}")
                        try:
                            sim_state,task_name = action_provider.load_data(data_json_list[data_idx])
                            if task_name!=args_cli.task:
                                raise ValueError(f" The {task_name} in the dataset is different from the {args_cli.task} being executed .")
                        except Exception as e:
                            print(f"Failed to load data: {e}")
                            raise e
                        try:
                            env.reset_to(sim_state, torch.tensor([0], device=env.device), is_relative=True)
                            env.sim.reset()
                            time.sleep(1)
                            action_provider.start_replay()
                            data_idx+=1
                        except Exception as e:
                            print(f"Failed to start replay: {e}")
                            raise e
                # print(f"env_state: {env_state}")
                # calculate instantaneous loop time
                loop_dt = current_time - last_loop_time
                last_loop_time = current_time
                recent_loop_times.append(loop_dt)
                
                # keep recent 100 loop times
                if len(recent_loop_times) > 100:
                    recent_loop_times.pop(0)
                
                # execute control step (in main thread, support rendering)
                controller.step()

                # print statistics and loop frequency periodically
                if current_time - last_stats_time >= args_cli.stats_interval:
                    # calculate while loop execution frequency
                    elapsed_time = current_time - loop_start_time
                    loop_frequency = loop_count / elapsed_time if elapsed_time > 0 else 0
                    
                    # calculate moving average frequency (based on recent loop times)
                    if recent_loop_times:
                        avg_loop_time = sum(recent_loop_times) / len(recent_loop_times)
                        moving_avg_frequency = 1.0 / avg_loop_time if avg_loop_time > 0 else 0
                        min_loop_time = min(recent_loop_times)
                        max_loop_time = max(recent_loop_times)
                        max_freq = 1.0 / min_loop_time if min_loop_time > 0 else 0
                        min_freq = 1.0 / max_loop_time if max_loop_time > 0 else 0
                    else:
                        moving_avg_frequency = 0
                        min_freq = max_freq = 0
                    
                    print(f"\n=== While loop execution frequency statistics ===")
                    print(f"loop execution count: {loop_count}")
                    print(f"running time: {elapsed_time:.2f} seconds")
                    print(f"overall average frequency: {loop_frequency:.2f} Hz")
                    print(f"moving average frequency: {moving_avg_frequency:.2f} Hz (last {len(recent_loop_times)} times)")
                    print(f"frequency range: {min_freq:.2f} - {max_freq:.2f} Hz")
                    print(f"average loop time: {(elapsed_time/loop_count*1000):.2f} ms")
                    if recent_loop_times:
                        print(f"recent loop time: {(avg_loop_time*1000):.2f} ms")
                    print(f"=============================")
                    
                    # print_stats(controller)
                    last_stats_time = current_time
       
                # check environment state
                if env.sim.is_stopped():
                    print("\nenvironment stopped")
                    break
                # rate_limiter.sleep(env)
    except KeyboardInterrupt:
        print("\nuser interrupted program")
    
    except Exception as e:
        print(f"\nprogram exception: {e}")
    
    finally:
        # 结束性能基准测试
        benchmark_result = performance_monitor.end_benchmark()
        if benchmark_result:
            logger.info(f"Benchmark completed: {benchmark_result.test_name}, "
                       f"duration: {benchmark_result.duration:.2f}s")

            # 输出性能统计
            perf_stats = performance_monitor.get_summary_stats()
            logger.info("Performance summary", extra={'performance_stats': perf_stats})

            # 资源清理
            resource_manager.cleanup_all()

        # 停止性能监控
        performance_monitor.stop_monitoring()

        # clean up resources
        print("\nclean up resources...")
        controller.cleanup()
        if 'image_server' in locals():
            image_server.stop()
        env.close()
        print("cleanup completed")
    # profiler.disable()
    # s = io.StringIO()
    # ps = pstats.Stats(profiler, stream=s).strip_dirs().sort_stats("time")
    # ps.print_stats(30)  

    # print(s.getvalue())

if __name__ == "__main__":
    try:
        main()
    finally:
        print("Performing final cleanup...")
        
        # Get current process information
        import os
        import subprocess
        import signal
        import time
        
        current_pid = os.getpid()
        print(f"Current main process PID: {current_pid}")
        
        try:
            # Find all related Python processes
            result = subprocess.run(['pgrep', '-f', 'sim_main.py'], 
                                  capture_output=True, text=True)
            if result.returncode == 0:
                pids = result.stdout.strip().split('\n')
                print(f"Found related processes: {pids}")
                
                for pid in pids:
                    if pid and pid != str(current_pid):
                        try:
                            print(f"Terminating child process: {pid}")
                            os.kill(int(pid), signal.SIGTERM)
                        except ProcessLookupError:
                            print(f"Process {pid} does not exist")
                        except Exception as e:
                            print(f"Failed to terminate process {pid}: {e}")
                
                # Wait for processes to exit
                time.sleep(2)
                
                # Check if there are any remaining processes, force kill them
                result2 = subprocess.run(['pgrep', '-f', 'sim_main.py'], 
                                       capture_output=True, text=True)
                if result2.returncode == 0:
                    remaining_pids = result2.stdout.strip().split('\n')
                    for pid in remaining_pids:
                        if pid and pid != str(current_pid):
                            try:
                                print(f"Force killing process: {pid}")
                                os.kill(int(pid), signal.SIGKILL)
                            except Exception as e:
                                print(f"Failed to force kill process {pid}: {e}")
                                
        except Exception as e:
            print(f"Error during process cleanup: {e}")
        
        try:
            simulation_app.close()
        except Exception as e:
            print(f"Failed to close simulation application: {e}")
            
        print("Program exit completed")
        
        # Force exit
        os._exit(0)

# python sim_main.py --device cpu  --enable_cameras  --task  Isaac-PickPlace-Cylinder-G129-Dex1-Joint   --enable_dex1_dds --robot_type g129
# python sim_main.py --device cpu  --enable_cameras  --task Isaac-PickPlace-Cylinder-G129-Dex3-Joint    --enable_dex3_dds --robot_type g129
# python sim_main.py --device cpu  --enable_cameras  --task Isaac-PickPlace-Cylinder-G129-Inspire-Joint    --enable_inspire_dds --robot_type g129

# python sim_main.py --device cpu  --enable_cameras  --task Isaac-PickPlace-RedBlock-G129-Dex1-Joint     --enable_dex1_dds --robot_type g129
# python sim_main.py --device cpu  --enable_cameras  --task Isaac-PickPlace-RedBlock-G129-Dex3-Joint    --enable_dex3_dds --robot_type g129
# python sim_main.py --device cpu  --enable_cameras  --task  Isaac-PickPlace-RedBlock-G129-Inspire-Joint    --enable_inspire_dds --robot_type g129


# python sim_main.py --device cpu  --enable_cameras  --task Isaac-Stack-RgyBlock-G129-Dex1-Joint     --enable_dex1_dds --robot_type g129
# python sim_main.py --device cpu  --enable_cameras  --task Isaac-Stack-RgyBlock-G129-Dex3-Joint     --enable_dex3_dds --robot_type g129
# python sim_main.py --device cpu  --enable_cameras  --task Isaac-Stack-RgyBlock-G129-Inspire-Joint     --enable_inspire_dds --robot_type g129




# python sim_main.py --device cpu  --enable_cameras  --task Isaac-Move-Cylinder-G129-Dex1-Wholebody  --robot_type g129 --enable_dex1_dds 
# python sim_main.py --device cpu  --enable_cameras  --task Isaac-Move-Cylinder-G129-Dex3-Wholebody  --robot_type g129 --enable_dex3_dds 
# python sim_main.py --device cpu  --enable_cameras  --task Isaac-Move-Cylinder-G129-Inspire-Wholebody  --robot_type g129 --enable_inspire_dds 


# python sim_main.py --device cpu  --enable_cameras  --task Isaac-PickPlace-Cylinder-H12-27dof-Inspire-Joint  --enable_inspire_dds --robot_type h1_2
# python sim_main.py --device cpu  --enable_cameras  --task Isaac-PickPlace-RedBlock-H12-27dof-Inspire-Joint  --enable_inspire_dds --robot_type h1_2
# python sim_main.py --device cpu  --enable_cameras  --task Isaac-Stack-RgyBlock-H12-27dof-Inspire-Joint --enable_inspire_dds --robot_type h1_2
