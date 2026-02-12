# Copyright (c) 2025, Unitree Robotics Co., Ltd. All Rights Reserved.
# License: Apache License, Version 2.0

"""G1 hand grasp RL task - Stage 2 of two-stage pipeline.

This task trains a hand-only grasping policy while the arm is frozen
at a pre-grasp position (either fixed or from a trained reach policy).
"""

import gymnasium as gym

from .hand_grasp_env_cfg import (
    HandGraspEnvCfg,
    HandGraspEnvCfg_PLAY,
)


gym.register(
    id="Isaac-G1-HandGrasp-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={"env_cfg_entry_point": HandGraspEnvCfg},
    disable_env_checker=True,
)

gym.register(
    id="Isaac-G1-HandGrasp-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={"env_cfg_entry_point": HandGraspEnvCfg_PLAY},
    disable_env_checker=True,
)
