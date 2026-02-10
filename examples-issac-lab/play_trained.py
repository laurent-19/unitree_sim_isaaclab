# play_trained.py
import argparse
import torch
from isaaclab.app import AppLauncher

# --- launch Isaac Sim / Kit first ---
parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
parser.add_argument("--checkpoint", type=str, required=True, help="Path to .pt checkpoint")
parser.add_argument("--steps", type=int, default=2000)
parser.add_argument(
    "--goal",
    type=float,
    nargs=3,
    default=None,
    metavar=("X", "Y", "Z"),
    help="Fixed ee goal in ROBOT BASE frame (x y z). If omitted, uses random goals from CommandsCfg.",
)

args = parser.parse_args()

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

# --- safe imports now ---
from isaaclab.envs import ManagerBasedRLEnv
from point_nav_env_cfg import G1ReachPoseEnvCfg_PLAY

try:
    from isaaclab.envs.wrappers.rsl_rl import RslRlVecEnvWrapper
except Exception:
    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper  # fallback for some installs

from rsl_rl.runners.on_policy_runner import OnPolicyRunner


def make_train_cfg(max_iters: int = 1):
    # MUST match training config (net sizes etc.)
    return {
        "seed": 42,
        "num_steps_per_env": 32,
        "max_iterations": max_iters,
        "save_interval": 200,
        "experiment_name": "g1_reach_chips",
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
            "value_loss_coef": 1.0,
            "max_grad_norm": 1.0,
        },
        "policy": {
            "class_name": "ActorCritic",
            "init_noise_std": 1.0,
            "actor_hidden_dims": [256, 256],
            "critic_hidden_dims": [256, 256],
            "activation": "elu",
        },
    }


def _unwrap_obs(x):
    # reset() often returns (obs, info)
    if isinstance(x, tuple):
        return x[0]
    return x


def _as_policy_input(obs, device):
    """Return something the ActorCritic expects: a mapping with key 'policy' -> Tensor."""
    obs = _unwrap_obs(obs)

    # TensorDict / dict-like (has keys + __getitem__)
    if hasattr(obs, "keys") and hasattr(obs, "__getitem__") and not torch.is_tensor(obs):
        # If it ALREADY has "policy", perfect: pass it through
        try:
            if "policy" in obs.keys():
                # ensure tensor is on correct device if needed
                pol = obs["policy"]
                if torch.is_tensor(pol) and pol.device != device:
                    obs["policy"] = pol.to(device)
                return obs
        except Exception:
            pass

        # Otherwise, wrap first entry as policy
        keys = list(obs.keys())
        pol = obs[keys[0]]
        if not torch.is_tensor(pol):
            pol = torch.as_tensor(pol, device=device, dtype=torch.float32)
        else:
            pol = pol.to(device)
        return {"policy": pol}

    # Raw tensor/array -> wrap
    if not torch.is_tensor(obs):
        obs = torch.as_tensor(obs, device=device, dtype=torch.float32)
    else:
        obs = obs.to(device)
    return {"policy": obs}


def _unpack_step(step_out):
    # supports both 4 and 5 return styles
    if len(step_out) == 4:
        obs, rew, done, info = step_out
        return obs, rew, done, info
    elif len(step_out) == 5:
        obs, rew, terminated, truncated, info = step_out
        done = terminated | truncated
        return obs, rew, done, info
    else:
        raise RuntimeError(f"Unexpected step() return length: {len(step_out)}")


def _get_base_env(wrapped_env):
    """Try to peel wrappers until we reach the IsaacLab env that has .scene.sensors."""
    e = wrapped_env
    seen = set()
    # peel a few common wrapper attribute names
    for _ in range(6):
        if id(e) in seen:
            break
        seen.add(id(e))

        if hasattr(e, "unwrapped"):
            try:
                ue = e.unwrapped
                if ue is not None and ue is not e:
                    e = ue
                    continue
            except Exception:
                pass

        for attr in ("env", "_env", "_wrapped_env"):
            if hasattr(e, attr):
                ne = getattr(e, attr)
                if ne is not None and ne is not e:
                    e = ne
                    break
        else:
            break
    return e


def main():
    cfg = G1ReachPoseEnvCfg_PLAY()

    # If user provided a fixed goal, force the command sampler to always pick that exact point
    if args.goal is not None:
        x, y, z = args.goal
        cfg.commands.ee_pose.ranges.pos_x = (x, x)
        cfg.commands.ee_pose.ranges.pos_y = (y, y)
        cfg.commands.ee_pose.ranges.pos_z = (z, z)

        # (optional) keep the goal from resampling mid-episode
        cfg.commands.ee_pose.resampling_time_range = (1e9, 1e9)

    env = ManagerBasedRLEnv(cfg=cfg)
    env = RslRlVecEnvWrapper(env)

    # For force printing we need the underlying env with .scene.sensors
    base_env = _get_base_env(env)

    # Contact sensor names (must match what you defined in point_nav_env_cfg.py)
    CYL_SENSORS = [
        "left_palm_cyl_contact",
        "left_zero_cyl_contact",
        "left_one_cyl_contact",
        "left_two_cyl_contact",
        "left_three_cyl_contact",
        "left_four_cyl_contact",
        "left_five_cyl_contact",
        "left_six_cyl_contact",
    ]

    runner = OnPolicyRunner(env, make_train_cfg(), log_dir=".", device=env.device)
    runner.load(args.checkpoint)
    policy = runner.get_inference_policy(device=env.device)

    obs = env.reset()
    obs_in = _as_policy_input(obs, env.device)

    step_i = 0
    for _ in range(args.steps):
        with torch.inference_mode():
            actions = policy(obs_in)

        step_out = env.step(actions)
        obs, rew, done, info = _unpack_step(step_out)

        # -------- FORCE PRINTS (RIGHT AFTER STEP) --------
        # prints every 10 sim steps to avoid spam
        if step_i % 10 == 0:
            if hasattr(base_env, "scene") and hasattr(base_env.scene, "sensors"):
                sensors = base_env.scene.sensors
                env_id = 0  # PLAY config is 1 env, but keep this explicit
                for name in CYL_SENSORS:
                    if name not in sensors:
                        continue
                    s = sensors[name]
                    f = s.data.net_forces_w  # typically (num_envs, M, 3)
                    try:
                        f0 = f[env_id].reshape(-1, 3)
                        maxF = (
                            torch.linalg.norm(f0, dim=-1).max().item()
                            if f0.numel() > 0
                            else 0.0
                        )
                    except Exception:
                        # very defensive fallback
                        maxF = float("nan")
                    print(f"{name}: max|F| = {maxF:.3f}")
            else:
                # If you see this, the wrapper unwrapping failed (rare)
                print("[warn] Could not access base_env.scene.sensors for force printing.")
        step_i += 1
        # -----------------------------------------------

        obs_in = _as_policy_input(obs, env.device)

        if torch.any(done):
            obs_in = _as_policy_input(env.reset(), env.device)

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
