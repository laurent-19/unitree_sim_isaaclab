#!/usr/bin/env python3
# Copyright (c) 2025, Unitree Robotics Co., Ltd. All Rights Reserved.
# License: Apache License, Version 2.0

"""Training script for G1 hand grasp RL task - Stage 2 of two-stage pipeline.

This script trains a hand-only grasping policy where the arm is frozen
at a pre-grasp position. Use this after training the reach policy (Stage 1).

Usage:
    # Train hand grasp policy
    python train_hand_grasp_rl.py --num_envs 512 --max_iterations 2000

    # Resume from checkpoint
    python train_hand_grasp_rl.py --checkpoint logs/g1_hand_grasp/model_1000.pt --resume
"""

import argparse
import os

# Set project root
project_root = os.path.dirname(os.path.abspath(__file__))
os.environ["PROJECT_ROOT"] = project_root

from isaaclab.app import AppLauncher

# Parse args
parser = argparse.ArgumentParser(description="Train G1 hand grasp policy (Stage 2)")
AppLauncher.add_app_launcher_args(parser)
parser.add_argument("--num_envs", type=int, default=512, help="Number of environments")
parser.add_argument("--max_iterations", type=int, default=2000, help="Max training iterations")
parser.add_argument("--logdir", type=str, default="logs/g1_hand_grasp", help="Log directory")
parser.add_argument("--checkpoint", type=str, default=None, help="Checkpoint to resume from")
parser.add_argument("--resume", action="store_true", help="Resume training from checkpoint")
parser.add_argument("--seed", type=int, default=42, help="Random seed")
args = parser.parse_args()

# Launch app
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

# Now safe to import
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


# Create package stubs to avoid triggering auto-importer
create_package_stub("tasks")
create_package_stub("tasks.common_config")
create_package_stub("tasks.g1_tasks")
create_package_stub("tasks.g1_tasks.hand_grasp_rl")

# Load common_config modules directly
robot_configs_path = os.path.join(project_root, "tasks/common_config/robot_configs.py")
robot_configs = load_module_direct("tasks.common_config.robot_configs", robot_configs_path)

camera_configs_path = os.path.join(project_root, "tasks/common_config/camera_configs.py")
camera_configs = load_module_direct("tasks.common_config.camera_configs", camera_configs_path)

# Patch common_config module with the exports
common_config = sys.modules["tasks.common_config"]
common_config.G1RobotPresets = robot_configs.G1RobotPresets
common_config.RobotJointTemplates = robot_configs.RobotJointTemplates
common_config.CameraPresets = camera_configs.CameraPresets

# Load hand_grasp_rl modules directly
grasp_mdp_path = os.path.join(project_root, "tasks/g1_tasks/hand_grasp_rl/grasp_mdp.py")
grasp_mdp = load_module_direct("tasks.g1_tasks.hand_grasp_rl.grasp_mdp", grasp_mdp_path)

# Patch the hand_grasp_rl package
hand_grasp_pkg = sys.modules["tasks.g1_tasks.hand_grasp_rl"]
hand_grasp_pkg.grasp_mdp = grasp_mdp

hand_grasp_env_cfg_path = os.path.join(project_root, "tasks/g1_tasks/hand_grasp_rl/hand_grasp_env_cfg.py")
hand_grasp_env_cfg = load_module_direct("tasks.g1_tasks.hand_grasp_rl.hand_grasp_env_cfg", hand_grasp_env_cfg_path)

HandGraspEnvCfg = hand_grasp_env_cfg.HandGraspEnvCfg

# RSL-RL imports
try:
    from isaaclab.envs.wrappers.rsl_rl import RslRlVecEnvWrapper
except ImportError:
    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper

from rsl_rl.runners.on_policy_runner import OnPolicyRunner


def main():
    # Create env config
    env_cfg = HandGraspEnvCfg()
    env_cfg.scene.num_envs = args.num_envs

    # Create environment
    env = ManagerBasedRLEnv(cfg=env_cfg)
    env = RslRlVecEnvWrapper(env)

    # Get dimensions
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]

    print(f"\n{'='*60}")
    print("HAND GRASP RL - STAGE 2 TRAINING")
    print("="*60)
    print(f"  Observation dim: {obs_dim}")
    print(f"  Action dim: {act_dim}")
    print(f"  Environments: {args.num_envs}")
    print(f"  Max iterations: {args.max_iterations}")
    print("="*60 + "\n")

    # Training config (smaller network for simpler task)
    train_cfg = {
        "seed": args.seed,
        "num_steps_per_env": 32,
        "max_iterations": args.max_iterations,
        "save_interval": 200,
        "experiment_name": "g1_hand_grasp",
        "empirical_normalization": False,
        "run_name": "",
        "logger": "tensorboard",
        "neptune_project": "isaaclab",
        "wandb_project": "isaaclab",
        "resume": False,
        "load_run": -1,
        "load_checkpoint": -1,

        "obs_groups": {
            "policy": ["policy"],
            "critic": ["policy"],
        },

        "algorithm": {
            "class_name": "PPO",
            "learning_rate": 3e-4,
            "num_learning_epochs": 5,
            "num_mini_batches": 4,
            "gamma": 0.99,
            "lam": 0.95,
            "clip_param": 0.2,
            "entropy_coef": 0.01,
            "value_loss_coef": 1.0,
            "max_grad_norm": 1.0,
            "schedule": "adaptive",
            "desired_kl": 0.01,
        },

        "policy": {
            "class_name": "ActorCritic",
            "init_noise_std": 0.5,  # Lower for hand precision
            "actor_hidden_dims": [128, 128],  # Smaller network for simpler task
            "critic_hidden_dims": [128, 128],
            "activation": "elu",
        },
    }

    # Determine log directory
    log_dir = args.logdir
    if args.resume and args.checkpoint is not None:
        log_dir = os.path.dirname(os.path.abspath(args.checkpoint))

    print(f"Log directory: {log_dir}")

    # Create runner
    runner = OnPolicyRunner(env, train_cfg, log_dir=log_dir, device=env.device)

    # Load checkpoint if provided
    if args.checkpoint is not None:
        print(f"Loading checkpoint: {args.checkpoint}")
        runner.load(args.checkpoint)

    # Train
    print("Starting training...")
    runner.learn(num_learning_iterations=args.max_iterations, init_at_random_ep_len=True)

    env.close()
    print("Training complete!")


if __name__ == "__main__":
    main()
    simulation_app.close()
