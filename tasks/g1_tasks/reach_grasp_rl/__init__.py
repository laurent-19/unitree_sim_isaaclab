# Copyright (c) 2025, Unitree Robotics Co., Ltd. All Rights Reserved.
# License: Apache License, Version 2.0

"""G1 right arm reach and grasp RL task."""

import gymnasium as gym

from .reach_grasp_env_cfg import G1ReachGraspEnvCfg, G1ReachGraspEnvCfg_PLAY


gym.register(
    id="Isaac-G1-ReachGrasp-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={"env_cfg_entry_point": G1ReachGraspEnvCfg},
    disable_env_checker=True,
)

gym.register(
    id="Isaac-G1-ReachGrasp-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={"env_cfg_entry_point": G1ReachGraspEnvCfg_PLAY},
    disable_env_checker=True,
)
