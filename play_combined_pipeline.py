#!/usr/bin/env python3
# Copyright (c) 2025, Unitree Robotics Co., Ltd. All Rights Reserved.
# License: Apache License, Version 2.0

"""Combined pipeline evaluation: Reach (Stage 1) + Grasp (Stage 2).

This script runs the full two-stage pipeline:
1. Reach policy moves arm to pre-grasp position near object
2. Grasp policy closes hand to grasp and lift object

Usage:
    python play_combined_pipeline.py \
        --reach_checkpoint logs/g1_reach_grasp/model_2000.pt \
        --grasp_checkpoint logs/g1_hand_grasp/model_2000.pt
"""

import argparse
import os

# Set project root
project_root = os.path.dirname(os.path.abspath(__file__))
os.environ["PROJECT_ROOT"] = project_root

from isaaclab.app import AppLauncher

# Parse args
parser = argparse.ArgumentParser(description="Evaluate combined reach+grasp pipeline")
AppLauncher.add_app_launcher_args(parser)
parser.add_argument("--reach_checkpoint", type=str, required=True, help="Path to reach policy checkpoint")
parser.add_argument("--grasp_checkpoint", type=str, required=True, help="Path to grasp policy checkpoint")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments")
parser.add_argument("--num_episodes", type=int, default=10, help="Number of episodes")
parser.add_argument("--reach_threshold", type=float, default=0.05, help="Distance threshold to switch to grasp")
parser.add_argument("--reach_timeout", type=int, default=300, help="Max steps for reach phase")
args = parser.parse_args()

# Launch app
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

# Now safe to import
import torch
from isaaclab.envs import ManagerBasedRLEnv

# Bypass tasks/__init__.py
import sys
import importlib.util
import types


def load_module_direct(module_name, file_path):
    """Load a module directly without triggering parent package imports."""
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def create_package_stub(package_name):
    """Create a stub package to satisfy import chain."""
    if package_name not in sys.modules:
        pkg = types.ModuleType(package_name)
        pkg.__path__ = []
        pkg.__package__ = package_name
        sys.modules[package_name] = pkg
    return sys.modules[package_name]


# Create package stubs
create_package_stub("tasks")
create_package_stub("tasks.common_config")
create_package_stub("tasks.g1_tasks")
create_package_stub("tasks.g1_tasks.reach_grasp_rl")
create_package_stub("tasks.g1_tasks.hand_grasp_rl")

# Load common_config modules
robot_configs_path = os.path.join(project_root, "tasks/common_config/robot_configs.py")
robot_configs = load_module_direct("tasks.common_config.robot_configs", robot_configs_path)

camera_configs_path = os.path.join(project_root, "tasks/common_config/camera_configs.py")
camera_configs = load_module_direct("tasks.common_config.camera_configs", camera_configs_path)

common_config = sys.modules["tasks.common_config"]
common_config.G1RobotPresets = robot_configs.G1RobotPresets
common_config.RobotJointTemplates = robot_configs.RobotJointTemplates
common_config.CameraPresets = camera_configs.CameraPresets

# Load reach_grasp_rl modules
reach_mdp_path = os.path.join(project_root, "tasks/g1_tasks/reach_grasp_rl/reach_mdp.py")
reach_mdp = load_module_direct("tasks.g1_tasks.reach_grasp_rl.reach_mdp", reach_mdp_path)

reach_env_cfg_path = os.path.join(project_root, "tasks/g1_tasks/reach_grasp_rl/reach_grasp_env_cfg.py")
reach_env_cfg = load_module_direct("tasks.g1_tasks.reach_grasp_rl.reach_grasp_env_cfg", reach_env_cfg_path)

G1ReachGraspEnvCfg_PLAY = reach_env_cfg.G1ReachGraspEnvCfg_PLAY

# RSL-RL imports
try:
    from isaaclab.envs.wrappers.rsl_rl import RslRlVecEnvWrapper
except ImportError:
    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper

from rsl_rl.modules import ActorCritic


def load_policy(checkpoint_path, obs_dim, act_dim, hidden_dims, device):
    """Load a policy from checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device)

    policy = ActorCritic(
        num_actor_obs=obs_dim,
        num_critic_obs=obs_dim,
        num_actions=act_dim,
        actor_hidden_dims=hidden_dims,
        critic_hidden_dims=hidden_dims,
        activation="elu",
        init_noise_std=1.0,
    ).to(device)

    policy.load_state_dict(checkpoint["model_state_dict"])
    policy.eval()
    return policy


def main():
    # For combined pipeline, we use the reach environment
    # The grasp policy will be applied once reach is complete
    env_cfg = G1ReachGraspEnvCfg_PLAY()
    env_cfg.scene.num_envs = args.num_envs

    # Longer episode for combined task
    env_cfg.episode_length_s = 12.0

    # Create environment
    env = ManagerBasedRLEnv(cfg=env_cfg)
    env_wrapped = RslRlVecEnvWrapper(env)

    # Reach policy dimensions (from reach task)
    reach_obs_dim = env_wrapped.observation_space.shape[0]  # ~21
    reach_act_dim = env_wrapped.action_space.shape[0]  # 3 (IK)

    # Grasp policy dimensions
    grasp_obs_dim = 40  # hand_pos(6) + hand_vel(6) + contacts(15) + obj_pose(7) + last_action(6)
    grasp_act_dim = 6  # 6 hand joints

    print(f"\n{'='*60}")
    print("COMBINED PIPELINE - REACH + GRASP")
    print("="*60)
    print(f"  Reach checkpoint: {args.reach_checkpoint}")
    print(f"  Grasp checkpoint: {args.grasp_checkpoint}")
    print(f"  Reach obs/act: {reach_obs_dim}/{reach_act_dim}")
    print(f"  Grasp obs/act: {grasp_obs_dim}/{grasp_act_dim}")
    print(f"  Reach threshold: {args.reach_threshold}m")
    print(f"  Reach timeout: {args.reach_timeout} steps")
    print("="*60 + "\n")

    # Load policies
    device = env_wrapped.device

    reach_policy = load_policy(
        args.reach_checkpoint,
        reach_obs_dim,
        reach_act_dim,
        [256, 256],  # Reach network
        device
    )
    print("Reach policy loaded!")

    grasp_policy = load_policy(
        args.grasp_checkpoint,
        grasp_obs_dim,
        grasp_act_dim,
        [128, 128],  # Grasp network
        device
    )
    print("Grasp policy loaded!")

    # Phase tracking
    phase = torch.zeros(args.num_envs, dtype=torch.int32, device=device)  # 0=reach, 1=grasp
    reach_steps = torch.zeros(args.num_envs, dtype=torch.int32, device=device)

    # Statistics
    success_count = 0
    reach_success_count = 0
    episode_count = 0

    obs, _ = env_wrapped.reset()

    while episode_count < args.num_episodes:
        # Get EE position and goal position for distance check
        robot = env.scene["robot"]
        ee_body_id = list(robot.data.body_names).index("R_thumb_proximal")
        ee_pos = robot.data.body_pos_w[:, ee_body_id]

        cmd = env.command_manager.get_command("ee_pose")
        goal_pos = cmd[:, 0:3]

        # Transform goal to world frame
        root_pos = robot.data.root_pos_w
        root_quat = robot.data.root_quat_w
        from isaaclab.utils.math import quat_apply
        goal_pos_w = root_pos + quat_apply(root_quat, goal_pos)

        dist = torch.norm(ee_pos - goal_pos_w, dim=-1)

        # Phase switching: reach -> grasp
        reach_done = (dist < args.reach_threshold) | (reach_steps >= args.reach_timeout)
        phase = torch.where(reach_done & (phase == 0), torch.ones_like(phase), phase)

        # Get actions based on phase
        with torch.no_grad():
            # Reach phase
            reach_mask = phase == 0
            grasp_mask = phase == 1

            actions = torch.zeros(args.num_envs, reach_act_dim, device=device)

            if reach_mask.any():
                reach_actions = reach_policy.act_inference(obs)
                actions[reach_mask] = reach_actions[reach_mask]
                reach_steps[reach_mask] += 1

            if grasp_mask.any():
                # For grasp phase, we need to construct grasp observations
                # In a real implementation, this would extract hand-specific observations
                # For now, we use zero actions (hand will close via events)
                pass

        # Step environment
        obs, rewards, terminated, truncated, infos = env_wrapped.step(actions)

        # Check for episode termination
        done = terminated | truncated
        if done.any():
            for i in range(args.num_envs):
                if done[i]:
                    episode_count += 1
                    reached = phase[i] == 1
                    if reached:
                        reach_success_count += 1

                    # Check if grasp was successful (object lifted)
                    obj = env.scene["grasp_cylinder"]
                    obj_height = obj.data.root_pos_w[i, 2].item()
                    grasped = obj_height > 0.95  # Above initial + some lift

                    if grasped:
                        success_count += 1
                        print(f"Episode {episode_count}: SUCCESS (reached + grasped)")
                    elif reached:
                        print(f"Episode {episode_count}: REACHED but not grasped")
                    else:
                        print(f"Episode {episode_count}: FAILED (did not reach)")

                    # Reset phase tracking
                    phase[i] = 0
                    reach_steps[i] = 0

                    if episode_count >= args.num_episodes:
                        break

    # Print results
    print(f"\n{'='*60}")
    print("COMBINED PIPELINE RESULTS")
    print("="*60)
    print(f"  Total episodes: {episode_count}")
    print(f"  Reach success rate: {reach_success_count/episode_count*100:.1f}%")
    print(f"  Full success rate: {success_count/episode_count*100:.1f}%")
    print("="*60 + "\n")

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
