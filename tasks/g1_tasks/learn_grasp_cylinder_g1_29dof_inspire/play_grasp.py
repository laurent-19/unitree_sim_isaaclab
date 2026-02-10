# play_grasp.py
# Copyright (c) 2025. All Rights Reserved.
# License: Apache License, Version 2.0
"""
Playback/evaluation script for Learn-to-Grasp task.

Runs a trained policy or random actions to visualize grasping behavior.

Usage:
    # With trained checkpoint
    python tasks/g1_tasks/learn_grasp_cylinder_g1_29dof_inspire/play_grasp.py --checkpoint logs/learn_grasp/model_2000.pt

    # Random actions (for testing environment)
    python tasks/g1_tasks/learn_grasp_cylinder_g1_29dof_inspire/play_grasp.py --random

    # Scripted test (fingers closed)
    python tasks/g1_tasks/learn_grasp_cylinder_g1_29dof_inspire/play_grasp.py --scripted
"""

import os
import sys

# Add project root and script directory to path
# This script is in: tasks/g1_tasks/learn_grasp_cylinder_g1_29dof_inspire/
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(os.path.dirname(script_dir)))
sys.path.insert(0, project_root)
sys.path.insert(0, script_dir)  # For local imports
os.environ["PROJECT_ROOT"] = project_root

import argparse

from isaaclab.app import AppLauncher

# Parse arguments before Isaac imports
parser = argparse.ArgumentParser(description="Play Learn-to-Grasp policy")
AppLauncher.add_app_launcher_args(parser)
parser.add_argument("--checkpoint", type=str, default=None,
                    help="Path to trained checkpoint (.pt file)")
parser.add_argument("--steps", type=int, default=1000,
                    help="Number of simulation steps to run")
parser.add_argument("--random", action="store_true",
                    help="Use random actions instead of policy")
parser.add_argument("--scripted", action="store_true",
                    help="Use scripted finger closing test")
args = parser.parse_args()

# Launch Isaac Sim (with rendering)
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

# Now safe to import
import torch

from isaaclab.envs import ManagerBasedRLEnv

# Import directly to avoid triggering tasks/__init__.py auto-importer
# which loads other packages with pinocchio dependency issues
from learn_grasp_env_cfg import LearnGraspCylinderEnvCfg_PLAY

# RSL-RL imports (only needed if using checkpoint)
try:
    from isaaclab.envs.wrappers.rsl_rl import RslRlVecEnvWrapper
except ImportError:
    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper

from rsl_rl.runners.on_policy_runner import OnPolicyRunner


def make_train_cfg() -> dict:
    """Training config must match the one used during training."""
    return {
        "seed": 42,
        "num_steps_per_env": 24,
        "max_iterations": 1,
        "save_interval": 200,
        "experiment_name": "learn_grasp",
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
            "entropy_coef": 0.005,
            "value_loss_coef": 1.0,
            "max_grad_norm": 1.0,
        },
        "policy": {
            "class_name": "ActorCritic",
            "init_noise_std": 0.8,
            "actor_hidden_dims": [128, 128],
            "critic_hidden_dims": [128, 128],
            "activation": "elu",
        },
    }


def _unwrap_obs(x):
    """Handle observation returns that may be tuples."""
    if isinstance(x, tuple):
        return x[0]
    return x


def _as_policy_input(obs, device):
    """Convert observation to policy input format."""
    obs = _unwrap_obs(obs)

    if hasattr(obs, "keys") and "policy" in obs.keys():
        pol = obs["policy"]
        if torch.is_tensor(pol) and pol.device != device:
            obs["policy"] = pol.to(device)
        return obs

    if hasattr(obs, "keys"):
        keys = list(obs.keys())
        pol = obs[keys[0]]
        if not torch.is_tensor(pol):
            pol = torch.as_tensor(pol, device=device, dtype=torch.float32)
        return {"policy": pol.to(device)}

    if not torch.is_tensor(obs):
        obs = torch.as_tensor(obs, device=device, dtype=torch.float32)
    return {"policy": obs.to(device)}


def _unpack_step(step_out):
    """Handle both 4 and 5 return value step() formats."""
    if len(step_out) == 4:
        obs, rew, done, info = step_out
        return obs, rew, done, info
    elif len(step_out) == 5:
        obs, rew, terminated, truncated, info = step_out
        done = terminated | truncated
        return obs, rew, done, info
    else:
        raise RuntimeError(f"Unexpected step() return length: {len(step_out)}")


def main():
    print("\n" + "=" * 60)
    print("Learn-to-Grasp Playback")
    print("=" * 60)

    # Create environment (1 env for visualization)
    env_cfg = LearnGraspCylinderEnvCfg_PLAY()
    env = ManagerBasedRLEnv(cfg=env_cfg)

    print(f"\nEnvironment created:")
    print(f"  - Observation space: {env.observation_space}")
    print(f"  - Action space: {env.action_space}")

    # Determine control mode
    policy = None
    if args.checkpoint is not None and not args.random and not args.scripted:
        print(f"\nLoading policy from: {args.checkpoint}")
        wrapped_env = RslRlVecEnvWrapper(env)
        runner = OnPolicyRunner(
            wrapped_env,
            make_train_cfg(),
            log_dir=".",
            device=wrapped_env.device,
        )
        runner.load(args.checkpoint)
        policy = runner.get_inference_policy(device=wrapped_env.device)
        env = wrapped_env
        print("Policy loaded successfully!")
    elif args.scripted:
        print("\nUsing scripted finger control (gradual close)")
    else:
        print("\nUsing random actions")

    # Reset environment
    obs = env.reset()
    if policy is not None:
        obs_in = _as_policy_input(obs, env.device)

    print(f"\nRunning for {args.steps} steps...")
    print("-" * 60)

    # Scripted finger control state
    scripted_finger_pos = 0.0  # Start open

    total_reward = 0.0
    step_count = 0

    for step_i in range(args.steps):
        # Get action
        if policy is not None:
            with torch.inference_mode():
                actions = policy(obs_in)
        elif args.scripted:
            # Scripted: gradually close fingers based on phase
            if hasattr(env, "_task_phase"):
                phase = env._task_phase[0].item() if hasattr(env._task_phase, '__getitem__') else 0
            elif hasattr(env, "unwrapped") and hasattr(env.unwrapped, "_task_phase"):
                phase = env.unwrapped._task_phase[0].item()
            else:
                phase = min(step_i // 50, 4)  # Estimate phase from step

            # Close during grasp phases (1-3), open during approach (0) and release (4)
            if phase in [1, 2, 3]:
                scripted_finger_pos = min(scripted_finger_pos + 0.02, 1.5)  # Close
            else:
                scripted_finger_pos = max(scripted_finger_pos - 0.03, 0.0)  # Open

            actions = torch.full(
                (1, 6),
                scripted_finger_pos,
                device=env.device,
                dtype=torch.float32,
            )
        else:
            # Random actions
            actions = torch.rand(1, 6, device=env.device) * 2.0 - 0.5

        # Step environment
        step_out = env.step(actions)
        obs, rew, done, info = _unpack_step(step_out)

        total_reward += rew.sum().item()
        step_count += 1

        # Print status every 50 steps
        if step_i % 50 == 0:
            # Get phase info
            phase_str = "?"
            if hasattr(env, "_task_phase"):
                phase = env._task_phase[0].item() if hasattr(env._task_phase, '__getitem__') else "?"
                phase_names = ["APPROACH", "GRASP", "LIFT", "MOVE", "RELEASE"]
                phase_str = phase_names[phase] if isinstance(phase, int) and phase < 5 else str(phase)
            elif hasattr(env, "unwrapped") and hasattr(env.unwrapped, "_task_phase"):
                phase = env.unwrapped._task_phase[0].item()
                phase_names = ["APPROACH", "GRASP", "LIFT", "MOVE", "RELEASE"]
                phase_str = phase_names[phase] if phase < 5 else str(phase)

            print(f"Step {step_i:4d} | Phase: {phase_str:8s} | Reward: {rew.sum().item():6.3f} | Total: {total_reward:8.3f}")

        # Prepare next observation
        if policy is not None:
            obs_in = _as_policy_input(obs, env.device)

        # Handle episode reset
        if torch.any(done):
            print(f"\n--- Episode ended at step {step_i} ---")
            print(f"Episode reward: {total_reward:.3f}")
            total_reward = 0.0

            obs = env.reset()
            if policy is not None:
                obs_in = _as_policy_input(obs, env.device)

    print("\n" + "=" * 60)
    print(f"Playback complete. Steps: {step_count}")
    print("=" * 60)

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
