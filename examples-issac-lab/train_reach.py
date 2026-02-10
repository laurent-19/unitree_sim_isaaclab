# train_reach.py
import argparse
from isaaclab.app import AppLauncher
import os

# Launch Kit first
parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
parser.add_argument("--max_iterations", type=int, default=2000)
parser.add_argument("--logdir", type=str, default="logs/g1_reach_chips")
parser.add_argument("--checkpoint", type=str, default=None,
                    help="Path to a checkpoint file like .../model_2000.pt")
parser.add_argument("--resume", action="store_true",
                    help="If set with --checkpoint, continue writing logs/saves into the checkpoint folder.")
args = parser.parse_args()

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

# Now safe imports
from isaaclab.envs import ManagerBasedRLEnv
#from point_nav_env_cfg import G1ReachChipsEnvCfg
from point_nav_env_cfg import G1ReachPoseEnvCfg
# IsaacLab wrapper name can differ by version — try both
try:
    from isaaclab.envs.wrappers.rsl_rl import RslRlVecEnvWrapper
except Exception:
    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper  # fallback for some installs

from rsl_rl.runners.on_policy_runner import OnPolicyRunner


def main():
    env = ManagerBasedRLEnv(cfg=G1ReachPoseEnvCfg())
    env = RslRlVecEnvWrapper(env)

    train_cfg = {
        "seed": 42,

        "num_steps_per_env": 32,
        "max_iterations": args.max_iterations,
        "save_interval": 200,
        "experiment_name": "g1_reach_chips",

        "obs_groups": {
            "policy": ["policy"],
            "critic": ["policy"],
        },

        "algorithm": {
            "class_name": "PPO",          # 👈 add this (prevents the next likely KeyError)
            "learning_rate": 3e-4,
            "num_learning_epochs": 5,
            "num_mini_batches": 4,
            "gamma": 0.99,
            "lam": 0.95,
            "clip_param": 0.2,
            "entropy_coef": 0.01,
            "value_loss_coef": 1.0,
            "max_grad_norm": 1.0,
        },

        "policy": {
            "class_name": "ActorCritic",  # 👈 THIS fixes your current crash
            "init_noise_std": 1.0,
            "actor_hidden_dims": [256, 256],
            "critic_hidden_dims": [256, 256],
            "activation": "elu",
        },
    }


     # Decide logging directory
    log_dir = args.logdir
    if args.resume and args.checkpoint is not None:
        log_dir = os.path.dirname(os.path.abspath(args.checkpoint))

    # Create runner
    runner = OnPolicyRunner(env, train_cfg, log_dir=log_dir, device=env.device)

    # Load checkpoint (if provided) BEFORE learning
    if args.checkpoint is not None:
        runner.load(args.checkpoint)

    # Train
    runner.learn(num_learning_iterations=args.max_iterations, init_at_random_ep_len=True)

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
