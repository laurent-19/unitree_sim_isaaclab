# train_grasp.py
# Copyright (c) 2025. All Rights Reserved.
# License: Apache License, Version 2.0
"""
Training script for Learn-to-Grasp task.

Trains a policy to control right hand finger joints for grasping,
while arm movement is handled by scripted events.

Usage:
    python tasks/g1_tasks/learn_grasp_cylinder_g1_29dof_inspire/train_grasp.py --max_iterations 2000
    python tasks/g1_tasks/learn_grasp_cylinder_g1_29dof_inspire/train_grasp.py --headless --max_iterations 5000
    python tasks/g1_tasks/learn_grasp_cylinder_g1_29dof_inspire/train_grasp.py --checkpoint logs/learn_grasp/model_1000.pt --resume
"""

import argparse
import os
import sys

# Add project root and script directory to path
# This script is in: tasks/g1_tasks/learn_grasp_cylinder_g1_29dof_inspire/
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(os.path.dirname(script_dir)))
sys.path.insert(0, project_root)
sys.path.insert(0, script_dir)  # For local imports
os.environ["PROJECT_ROOT"] = project_root

from isaaclab.app import AppLauncher

# Parse arguments before Isaac imports
parser = argparse.ArgumentParser(description="Train Learn-to-Grasp policy")
AppLauncher.add_app_launcher_args(parser)
parser.add_argument("--max_iterations", type=int, default=3000,
                    help="Maximum training iterations")
parser.add_argument("--logdir", type=str, default="logs/learn_grasp",
                    help="Log directory for checkpoints and tensorboard")
parser.add_argument("--checkpoint", type=str, default=None,
                    help="Path to checkpoint to resume from")
parser.add_argument("--resume", action="store_true",
                    help="Resume training in same log directory as checkpoint")
parser.add_argument("--num_envs", type=int, default=256,
                    help="Number of parallel environments")
args = parser.parse_args()

# Launch Isaac Sim
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

# Now safe to import Isaac/torch
import torch

from isaaclab.envs import ManagerBasedRLEnv

# Import directly to avoid triggering tasks/__init__.py auto-importer
# which loads other packages with pinocchio dependency issues
from learn_grasp_env_cfg import LearnGraspCylinderEnvCfg

# RSL-RL imports
try:
    from isaaclab.envs.wrappers.rsl_rl import RslRlVecEnvWrapper
except ImportError:
    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper

from rsl_rl.runners.on_policy_runner import OnPolicyRunner


def make_train_cfg(max_iterations: int = 3000) -> dict:
    """Create training configuration for PPO.

    Tuned for 6-DOF finger control task.
    """
    return {
        "seed": 42,

        # Rollout settings
        "num_steps_per_env": 24,  # Steps per env before update
        "max_iterations": max_iterations,
        "save_interval": 200,
        "experiment_name": "learn_grasp",

        # Observation groups
        "obs_groups": {
            "policy": ["policy"],
            "critic": ["policy"],
        },

        # PPO algorithm settings
        "algorithm": {
            "class_name": "PPO",
            "learning_rate": 3e-4,
            "num_learning_epochs": 5,
            "num_mini_batches": 4,
            "gamma": 0.99,
            "lam": 0.95,
            "clip_param": 0.2,
            "entropy_coef": 0.005,  # Lower entropy for precise finger control
            "value_loss_coef": 1.0,
            "max_grad_norm": 1.0,
        },

        # Actor-Critic network
        "policy": {
            "class_name": "ActorCritic",
            "init_noise_std": 0.8,  # Lower initial noise for finger control
            "actor_hidden_dims": [128, 128],  # Smaller network for 6-DOF
            "critic_hidden_dims": [128, 128],
            "activation": "elu",
        },
    }


def main():
    print("\n" + "=" * 60)
    print("Learn-to-Grasp Training")
    print("=" * 60)

    # Create environment config with specified num_envs
    env_cfg = LearnGraspCylinderEnvCfg()
    env_cfg.scene.num_envs = args.num_envs

    # Create environment
    env = ManagerBasedRLEnv(cfg=env_cfg)
    env = RslRlVecEnvWrapper(env)

    print(f"\nEnvironment created:")
    print(f"  - Number of envs: {env.num_envs}")
    print(f"  - Observation space: {env.observation_space}")
    print(f"  - Action space: {env.action_space}")

    # Determine log directory
    log_dir = args.logdir
    if args.resume and args.checkpoint is not None:
        log_dir = os.path.dirname(os.path.abspath(args.checkpoint))
        print(f"  - Resuming in: {log_dir}")

    # Create training config
    train_cfg = make_train_cfg(args.max_iterations)

    # Create runner
    runner = OnPolicyRunner(
        env,
        train_cfg,
        log_dir=log_dir,
        device=env.device,
    )

    # Load checkpoint if provided
    if args.checkpoint is not None:
        print(f"\nLoading checkpoint: {args.checkpoint}")
        runner.load(args.checkpoint)

    # Start training
    print(f"\nStarting training for {args.max_iterations} iterations...")
    print(f"Logs will be saved to: {log_dir}")
    print("-" * 60)

    runner.learn(
        num_learning_iterations=args.max_iterations,
        init_at_random_ep_len=True,
    )

    print("\nTraining complete!")
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
