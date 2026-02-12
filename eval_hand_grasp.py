#!/usr/bin/env python3
# Copyright (c) 2025, Unitree Robotics Co., Ltd. All Rights Reserved.
# License: Apache License, Version 2.0

"""Evaluation script for hand grasping policy.

Runs N episodes and reports grasp success metrics for paper experiments.

Usage:
    # Evaluate tactile policy
    python eval_hand_grasp.py --checkpoint logs/tactile/model_3000.pt --num_episodes 100

    # Evaluate no-tactile policy
    python eval_hand_grasp.py --checkpoint logs/no_tactile/model_3000.pt --num_episodes 100 --no_tactile

    # Save results to file
    python eval_hand_grasp.py --checkpoint logs/tactile/model_3000.pt --output results.json

    # Test with different objects (requires env config modification)
    python eval_hand_grasp.py --checkpoint logs/tactile/model_3000.pt --object_type cube
"""

import argparse
import json
import os
import time
from collections import defaultdict

from isaaclab.app import AppLauncher

# Parse arguments BEFORE Isaac Lab imports
parser = argparse.ArgumentParser(description="Evaluate hand grasping policy")
AppLauncher.add_app_launcher_args(parser)

# Evaluation arguments
parser.add_argument("--checkpoint", type=str, required=True,
                    help="Path to policy checkpoint (.pt file)")
parser.add_argument("--num_episodes", type=int, default=100,
                    help="Number of episodes to evaluate")
parser.add_argument("--output", type=str, default=None,
                    help="Output JSON file for results")
parser.add_argument("--no_tactile", action="store_true",
                    help="Use no-tactile environment config")
parser.add_argument("--num_envs", type=int, default=64,
                    help="Number of parallel environments for evaluation")
parser.add_argument("--seed", type=int, default=0,
                    help="Random seed for reproducibility")
parser.add_argument("--render", action="store_true",
                    help="Enable rendering during evaluation")

# Metric thresholds
parser.add_argument("--lift_threshold", type=float, default=0.90,
                    help="Height threshold for successful lift (m)")
parser.add_argument("--hold_steps", type=int, default=50,
                    help="Steps to hold for successful grasp")

args = parser.parse_args()

# Launch simulation
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

# Safe imports
import torch
import numpy as np
from isaaclab.envs import ManagerBasedRLEnv

# Import task config directly using importlib to avoid triggering full package scan
import sys
import importlib.util

_script_dir = os.path.dirname(os.path.abspath(__file__))
_env_cfg_path = os.path.join(_script_dir, "tasks", "g1_tasks", "hand_grasp_inspire", "hand_grasp_inspire_env_cfg.py")
_spec = importlib.util.spec_from_file_location("hand_grasp_inspire_env_cfg", _env_cfg_path)
_env_cfg_module = importlib.util.module_from_spec(_spec)
sys.modules["hand_grasp_inspire_env_cfg"] = _env_cfg_module
_spec.loader.exec_module(_env_cfg_module)

HandGraspInspireEnvCfg = _env_cfg_module.HandGraspInspireEnvCfg
HandGraspInspireNoTactileEnvCfg = _env_cfg_module.HandGraspInspireNoTactileEnvCfg

# RSL-RL imports
try:
    from isaaclab.envs.wrappers.rsl_rl import RslRlVecEnvWrapper
except ImportError:
    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper

from rsl_rl.runners.on_policy_runner import OnPolicyRunner


class EpisodeMetrics:
    """Track metrics for individual episodes."""

    def __init__(self, num_envs: int, device: torch.device):
        self.num_envs = num_envs
        self.device = device
        self.reset()

    def reset(self):
        """Reset metrics for new episodes."""
        self.episode_lengths = torch.zeros(self.num_envs, device=self.device)
        self.episode_rewards = torch.zeros(self.num_envs, device=self.device)
        self.max_lift_height = torch.zeros(self.num_envs, device=self.device)
        self.max_contacts = torch.zeros(self.num_envs, device=self.device)
        self.grasp_success = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        self.lift_success = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        self.hold_counter = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)

    def update(self, env, rewards: torch.Tensor, lift_threshold: float, hold_steps: int):
        """Update metrics after each step."""
        self.episode_lengths += 1
        self.episode_rewards += rewards

        # Get object height
        obj = env.unwrapped.scene["object"]
        obj_height = obj.data.root_pos_w[:, 2]
        self.max_lift_height = torch.maximum(self.max_lift_height, obj_height)

        # Check lift success
        lifted = obj_height > lift_threshold
        self.lift_success |= lifted

        # Track hold duration
        self.hold_counter = torch.where(
            lifted,
            self.hold_counter + 1,
            torch.zeros_like(self.hold_counter)
        )

        # Grasp success requires holding for N steps
        self.grasp_success |= (self.hold_counter >= hold_steps)

        # Count finger contacts
        if hasattr(env.unwrapped, "_get_fingertip_forces"):
            forces = env.unwrapped._get_fingertip_forces()
            contacts = (forces > 0.5).sum(dim=-1)
            self.max_contacts = torch.maximum(self.max_contacts, contacts.float())


def evaluate_policy(
    env,
    policy,
    num_episodes: int,
    lift_threshold: float = 0.90,
    hold_steps: int = 50,
) -> dict:
    """Run evaluation and collect metrics.

    Args:
        env: Wrapped environment
        policy: Policy to evaluate
        num_episodes: Number of episodes to run
        lift_threshold: Height threshold for lift success
        hold_steps: Steps to hold for grasp success

    Returns:
        Dictionary of evaluation metrics
    """
    device = env.device
    num_envs = env.num_envs

    # Results storage
    all_results = defaultdict(list)
    completed_episodes = 0

    # Episode metrics tracker
    metrics = EpisodeMetrics(num_envs, device)

    # Reset environment
    obs, _ = env.reset()

    print(f"[eval] Starting evaluation: {num_episodes} episodes across {num_envs} envs")
    start_time = time.time()

    while completed_episodes < num_episodes:
        # Get action from policy
        with torch.no_grad():
            actions = policy(obs)

        # Step environment
        obs, rewards, dones, truncated, info = env.step(actions)
        terminated = dones | truncated

        # Update metrics
        metrics.update(env, rewards, lift_threshold, hold_steps)

        # Process completed episodes
        done_envs = terminated.nonzero(as_tuple=True)[0]

        for env_idx in done_envs:
            if completed_episodes >= num_episodes:
                break

            # Record results
            all_results["episode_length"].append(metrics.episode_lengths[env_idx].item())
            all_results["episode_reward"].append(metrics.episode_rewards[env_idx].item())
            all_results["max_lift_height"].append(metrics.max_lift_height[env_idx].item())
            all_results["max_contacts"].append(metrics.max_contacts[env_idx].item())
            all_results["grasp_success"].append(metrics.grasp_success[env_idx].item())
            all_results["lift_success"].append(metrics.lift_success[env_idx].item())

            completed_episodes += 1

            # Progress report
            if completed_episodes % 10 == 0:
                elapsed = time.time() - start_time
                success_rate = np.mean(all_results["grasp_success"]) * 100
                print(f"[eval] Episodes: {completed_episodes}/{num_episodes}, "
                      f"Success: {success_rate:.1f}%, Time: {elapsed:.1f}s")

            # Reset metrics for this env
            metrics.episode_lengths[env_idx] = 0
            metrics.episode_rewards[env_idx] = 0
            metrics.max_lift_height[env_idx] = 0
            metrics.max_contacts[env_idx] = 0
            metrics.grasp_success[env_idx] = False
            metrics.lift_success[env_idx] = False
            metrics.hold_counter[env_idx] = 0

    # Compute summary statistics
    elapsed = time.time() - start_time
    summary = {
        "num_episodes": num_episodes,
        "eval_time_s": elapsed,
        "grasp_success_rate": np.mean(all_results["grasp_success"]),
        "lift_success_rate": np.mean(all_results["lift_success"]),
        "mean_episode_length": np.mean(all_results["episode_length"]),
        "std_episode_length": np.std(all_results["episode_length"]),
        "mean_episode_reward": np.mean(all_results["episode_reward"]),
        "std_episode_reward": np.std(all_results["episode_reward"]),
        "mean_max_lift_height": np.mean(all_results["max_lift_height"]),
        "mean_max_contacts": np.mean(all_results["max_contacts"]),
        "raw_results": dict(all_results),
    }

    return summary


def main():
    """Main evaluation function."""
    print(f"[eval] Loading checkpoint: {args.checkpoint}")

    # Verify checkpoint exists
    if not os.path.exists(args.checkpoint):
        print(f"[ERROR] Checkpoint not found: {args.checkpoint}")
        return

    # Select environment config
    if args.no_tactile:
        print("[eval] Using NO-TACTILE environment")
        env_cfg = HandGraspInspireNoTactileEnvCfg()
    else:
        print("[eval] Using TACTILE environment")
        env_cfg = HandGraspInspireEnvCfg()

    # Configure for evaluation
    env_cfg.scene.num_envs = args.num_envs

    # Create environment
    env = ManagerBasedRLEnv(cfg=env_cfg)
    env = RslRlVecEnvWrapper(env)

    print(f"[eval] Observation dim: {env.observation_space.shape[0]}")
    print(f"[eval] Action dim: {env.action_space.shape[0]}")

    # Load policy from checkpoint
    # RSL-RL stores policy in checkpoint dict
    checkpoint = torch.load(args.checkpoint, map_location=env.device)

    # Create dummy training config for runner
    train_cfg = {
        "seed": args.seed,
        "num_steps_per_env": 32,
        "max_iterations": 1,
        "save_interval": 1,
        "experiment_name": "eval",
        "obs_groups": {"policy": ["policy"], "critic": ["policy"]},
        "algorithm": {
            "class_name": "PPO",
            "learning_rate": 3e-4,
            "num_learning_epochs": 5,
            "num_mini_batches": 4,
            "gamma": 0.99,
            "lam": 0.95,
            "clip_param": 0.2,
            "entropy_coef": 0.01,
        },
        "policy": {
            "class_name": "ActorCritic",
            "init_noise_std": 1.0,
            "actor_hidden_dims": [256, 128, 64],
            "critic_hidden_dims": [256, 128, 64],
            "activation": "elu",
        },
    }

    # Create runner and load checkpoint
    runner = OnPolicyRunner(env, train_cfg, log_dir="/tmp/eval", device=env.device)
    runner.load(args.checkpoint)

    # Get inference policy
    policy = runner.get_inference_policy(device=env.device)

    # Run evaluation
    results = evaluate_policy(
        env=env,
        policy=policy,
        num_episodes=args.num_episodes,
        lift_threshold=args.lift_threshold,
        hold_steps=args.hold_steps,
    )

    # Print results
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Episodes: {results['num_episodes']}")
    print(f"Eval time: {results['eval_time_s']:.1f}s")
    print("-" * 60)
    print(f"Grasp Success Rate: {results['grasp_success_rate']*100:.1f}%")
    print(f"Lift Success Rate: {results['lift_success_rate']*100:.1f}%")
    print(f"Mean Episode Length: {results['mean_episode_length']:.1f} +/- {results['std_episode_length']:.1f}")
    print(f"Mean Episode Reward: {results['mean_episode_reward']:.2f} +/- {results['std_episode_reward']:.2f}")
    print(f"Mean Max Lift Height: {results['mean_max_lift_height']:.3f}m")
    print(f"Mean Max Contacts: {results['mean_max_contacts']:.1f}")
    print("=" * 60)

    # Save results to file
    if args.output:
        # Remove raw results for JSON (can be large)
        save_results = {k: v for k, v in results.items() if k != "raw_results"}
        save_results["checkpoint"] = args.checkpoint
        save_results["no_tactile"] = args.no_tactile
        save_results["lift_threshold"] = args.lift_threshold
        save_results["hold_steps"] = args.hold_steps

        with open(args.output, "w") as f:
            json.dump(save_results, f, indent=2)
        print(f"\n[eval] Results saved to: {args.output}")

    env.close()
    print("\n[eval] Evaluation complete!")


if __name__ == "__main__":
    main()
    simulation_app.close()
