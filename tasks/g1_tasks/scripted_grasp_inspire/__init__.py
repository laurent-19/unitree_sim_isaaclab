# Copyright (c) 2025, Unitree Robotics Co., Ltd. All Rights Reserved.
# License: Apache License, Version 2.0

"""Scripted arm grasp task for G1 robot with Inspire hand.

This task demonstrates autonomous grasping where the arm moves to a
pre-grasp position via scripted trajectory, then closes the hand.
No DDS input is required - use with --action_source scripted.
"""

import gymnasium as gym

from . import scripted_grasp_inspire_env_cfg


gym.register(
    id="Isaac-ScriptedGrasp-Inspire",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": scripted_grasp_inspire_env_cfg.ScriptedGraspInspireEnvCfg,
    },
    disable_env_checker=True,
)
