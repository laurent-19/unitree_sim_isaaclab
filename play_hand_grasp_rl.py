#!/usr/bin/env python3
# Copyright (c) 2025, Unitree Robotics Co., Ltd. All Rights Reserved.
# License: Apache License, Version 2.0

"""Evaluation script for G1 hand grasp RL policy - Stage 2 of two-stage pipeline.

Usage:
    # Play with trained checkpoint
    python play_hand_grasp_rl.py --checkpoint logs/g1_hand_grasp/model_2000.pt

    # Record video
    python play_hand_grasp_rl.py --checkpoint logs/g1_hand_grasp/model_2000.pt --video_path output.mp4
"""

import argparse
import os

# Set project root
project_root = os.path.dirname(os.path.abspath(__file__))
os.environ["PROJECT_ROOT"] = project_root

from isaaclab.app import AppLauncher

# Parse args
parser = argparse.ArgumentParser(description="Evaluate G1 hand grasp policy")
AppLauncher.add_app_launcher_args(parser)
parser.add_argument("--checkpoint", type=str, required=True, help="Path to policy checkpoint")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments")
parser.add_argument("--num_episodes", type=int, default=10, help="Number of episodes to run")
parser.add_argument("--video_path", type=str, default=None, help="Path to save video")
args = parser.parse_args()

# Launch app (no headless for visualization)
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

# Now safe to import
import torch
from isaaclab.envs import ManagerBasedRLEnv

# Bypass tasks/__init__.py auto-importer by loading modules directly
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

# Load hand_grasp_rl modules
grasp_mdp_path = os.path.join(project_root, "tasks/g1_tasks/hand_grasp_rl/grasp_mdp.py")
grasp_mdp = load_module_direct("tasks.g1_tasks.hand_grasp_rl.grasp_mdp", grasp_mdp_path)

hand_grasp_pkg = sys.modules["tasks.g1_tasks.hand_grasp_rl"]
hand_grasp_pkg.grasp_mdp = grasp_mdp

hand_grasp_env_cfg_path = os.path.join(project_root, "tasks/g1_tasks/hand_grasp_rl/hand_grasp_env_cfg.py")
hand_grasp_env_cfg = load_module_direct("tasks.g1_tasks.hand_grasp_rl.hand_grasp_env_cfg", hand_grasp_env_cfg_path)

HandGraspEnvCfg_PLAY = hand_grasp_env_cfg.HandGraspEnvCfg_PLAY

# RSL-RL imports
try:
    from isaaclab.envs.wrappers.rsl_rl import RslRlVecEnvWrapper
except ImportError:
    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper

from rsl_rl.modules import ActorCritic


def main():
    # Create play environment config
    env_cfg = HandGraspEnvCfg_PLAY()
    env_cfg.scene.num_envs = args.num_envs

    # Create environment
    env = ManagerBasedRLEnv(cfg=env_cfg)
    env = RslRlVecEnvWrapper(env)

    # Get dimensions
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]

    print(f"\n{'='*60}")
    print("HAND GRASP RL - EVALUATION")
    print("="*60)
    print(f"  Checkpoint: {args.checkpoint}")
    print(f"  Observation dim: {obs_dim}")
    print(f"  Action dim: {act_dim}")
    print(f"  Environments: {args.num_envs}")
    print(f"  Episodes: {args.num_episodes}")
    print("="*60 + "\n")

    # Load policy
    checkpoint = torch.load(args.checkpoint, map_location=env.device)

    # Create policy network (must match training config)
    policy = ActorCritic(
        num_actor_obs=obs_dim,
        num_critic_obs=obs_dim,
        num_actions=act_dim,
        actor_hidden_dims=[128, 128],
        critic_hidden_dims=[128, 128],
        activation="elu",
        init_noise_std=0.5,
    ).to(env.device)

    # Load weights
    policy.load_state_dict(checkpoint["model_state_dict"])
    policy.eval()

    print("Policy loaded successfully!")

    # Run evaluation
    success_count = 0
    drop_count = 0
    episode_count = 0
    total_rewards = []

    obs, _ = env.reset()
    episode_reward = torch.zeros(args.num_envs, device=env.device)

    while episode_count < args.num_episodes:
        # Get action from policy
        with torch.no_grad():
            actions = policy.act_inference(obs)

        # Step environment
        obs, rewards, terminated, truncated, infos = env.step(actions)
        episode_reward += rewards

        # Check for episode termination
        done = terminated | truncated
        if done.any():
            for i in range(args.num_envs):
                if done[i]:
                    episode_count += 1
                    total_rewards.append(episode_reward[i].item())

                    # Check termination reason
                    if "grasp_success" in infos and infos.get("grasp_success", torch.zeros(1))[i]:
                        success_count += 1
                        print(f"Episode {episode_count}: SUCCESS (reward={episode_reward[i]:.2f})")
                    elif "object_dropped" in infos and infos.get("object_dropped", torch.zeros(1))[i]:
                        drop_count += 1
                        print(f"Episode {episode_count}: DROPPED (reward={episode_reward[i]:.2f})")
                    else:
                        print(f"Episode {episode_count}: TIMEOUT (reward={episode_reward[i]:.2f})")

                    episode_reward[i] = 0

                    if episode_count >= args.num_episodes:
                        break

    # Print statistics
    print(f"\n{'='*60}")
    print("EVALUATION RESULTS")
    print("="*60)
    print(f"  Total episodes: {episode_count}")
    print(f"  Success rate: {success_count/episode_count*100:.1f}%")
    print(f"  Drop rate: {drop_count/episode_count*100:.1f}%")
    print(f"  Mean reward: {sum(total_rewards)/len(total_rewards):.2f}")
    print("="*60 + "\n")

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
