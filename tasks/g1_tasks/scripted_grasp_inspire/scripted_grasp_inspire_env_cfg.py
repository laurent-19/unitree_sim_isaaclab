# Copyright (c) 2025, Unitree Robotics Co., Ltd. All Rights Reserved.
# License: Apache License, Version 2.0

"""Environment configuration for scripted arm grasp task.

This task demonstrates autonomous grasping without requiring DDS input.
The robot arm moves to a pre-grasp position via scripted trajectory,
then the hand closes to grasp the cylinder.

NOTE: This configuration does NOT import pink.tasks to avoid pinocchio dependency.
"""

import torch

import isaaclab.envs.mdp as base_mdp
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.utils import configclass
from isaaclab.assets import ArticulationCfg, RigidObjectCfg
from isaaclab.sensors import ContactSensorCfg
import isaaclab.sim as sim_utils

from . import mdp

from tasks.common_config import G1RobotPresets, CameraPresets
from tasks.common_event.event_manager import SimpleEvent, SimpleEventManager
from tasks.common_scene.base_scene_pickplace_cylindercfg import TableCylinderSceneCfg


@configclass
class ScriptedGraspSceneCfg(TableCylinderSceneCfg):
    """Scene configuration for scripted grasp task.

    Inherits from TableCylinderSceneCfg but overrides the object position
    to be within arm reach for the scripted pre-grasp pose.
    """

    # Robot configuration - G1 with Inspire hand, fixed base
    robot: ArticulationCfg = G1RobotPresets.g1_29dof_inspire_base_fix()

    # Override cylinder position to be within RIGHT arm reach
    # Robot is at (-0.15, 0.0, 0.76) facing +Y direction
    # Place cylinder to the LEFT of right arm (closer to center) for easier reach
    object = RigidObjectCfg(
        prim_path="/World/envs/env_.*/Object",
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(-0.05, 0.25, 0.82),  # Slightly left of center, in front, reachable height
            rot=[1, 0, 0, 0]
        ),
        spawn=sim_utils.CylinderCfg(
            radius=0.025,    # Slightly larger for easier grasping
            height=0.12,     # Short cylinder
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.1),  # Light object
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.8, 0.2, 0.2),  # Red color
                metallic=0.5
            ),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=1.5,
                dynamic_friction=1.5,
                restitution=0.0,
            ),
        ),
    )

    # Tactile contact sensors for Inspire Hand (right hand only for this task)
    right_fingertip_contacts = ContactSensorCfg(
        prim_path="/World/envs/env_.*/Robot/R_(index|middle|ring|pinky)_intermediate",
        history_length=3,
        debug_vis=False,
    )
    right_thumb_contacts = ContactSensorCfg(
        prim_path="/World/envs/env_.*/Robot/R_thumb_distal",
        history_length=3,
        debug_vis=False,
    )
    right_finger_pad_contacts = ContactSensorCfg(
        prim_path="/World/envs/env_.*/Robot/R_(index|middle|ring|pinky)_proximal",
        history_length=3,
        debug_vis=False,
    )
    right_palm_contacts = ContactSensorCfg(
        prim_path="/World/envs/env_.*/Robot/R_thumb_proximal",
        history_length=3,
        debug_vis=False,
    )

    # Left hand contacts (optional, but included for completeness)
    left_fingertip_contacts = ContactSensorCfg(
        prim_path="/World/envs/env_.*/Robot/L_(index|middle|ring|pinky)_intermediate",
        history_length=3,
        debug_vis=False,
    )
    left_thumb_contacts = ContactSensorCfg(
        prim_path="/World/envs/env_.*/Robot/L_thumb_distal",
        history_length=3,
        debug_vis=False,
    )
    left_finger_pad_contacts = ContactSensorCfg(
        prim_path="/World/envs/env_.*/Robot/L_(index|middle|ring|pinky)_proximal",
        history_length=3,
        debug_vis=False,
    )
    left_palm_contacts = ContactSensorCfg(
        prim_path="/World/envs/env_.*/Robot/L_thumb_proximal",
        history_length=3,
        debug_vis=False,
    )

    # Camera configuration
    front_camera = CameraPresets.g1_front_camera()
    left_wrist_camera = CameraPresets.left_inspire_wrist_camera()
    right_wrist_camera = CameraPresets.right_inspire_wrist_camera()


@configclass
class ActionsCfg:
    """Action configuration using direct joint position control."""
    joint_pos = base_mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=[".*"],
        scale=1.0,
        use_default_offset=True
    )


@configclass
class ObservationsCfg:
    """Observation configuration."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Policy observation group."""
        robot_joint_state = ObsTerm(func=mdp.get_robot_boy_joint_states)
        robot_inspire_state = ObsTerm(func=mdp.get_robot_inspire_joint_states)
        robot_tactile_state = ObsTerm(func=mdp.get_inspire_tactile_state)
        camera_image = ObsTerm(func=mdp.get_camera_image)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = False

    policy: PolicyCfg = PolicyCfg()


@configclass
class TerminationsCfg:
    """Termination conditions."""
    success = DoneTerm(func=mdp.reset_object_estimate)


@configclass
class RewardsCfg:
    """Reward configuration."""
    reward = RewTerm(func=mdp.compute_reward, weight=1.0)


@configclass
class EventCfg:
    """Event configuration for object reset."""
    reset_object = EventTermCfg(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {
                "x": [-0.02, 0.02],
                "y": [-0.02, 0.02],
            },
            "velocity_range": {},
            "asset_cfg": SceneEntityCfg("object"),
        },
    )


@configclass
class ScriptedGraspInspireEnvCfg(ManagerBasedRLEnvCfg):
    """Environment configuration for scripted arm grasp task."""

    # Scene settings
    scene: ScriptedGraspSceneCfg = ScriptedGraspSceneCfg(
        num_envs=1,
        env_spacing=2.5,
        replicate_physics=True
    )

    # MDP settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events = EventCfg()
    rewards: RewardsCfg = RewardsCfg()

    # Disable unused managers
    commands = None
    curriculum = None

    def __post_init__(self):
        """Post initialization."""
        # General settings
        self.decimation = 2
        self.episode_length_s = 30.0  # Longer episode for scripted motion

        # Simulation settings
        self.sim.dt = 0.005
        self.sim.render_interval = self.decimation

        # Update contact sensor periods
        self.scene.right_fingertip_contacts.update_period = self.sim.dt
        self.scene.right_thumb_contacts.update_period = self.sim.dt
        self.scene.right_finger_pad_contacts.update_period = self.sim.dt
        self.scene.right_palm_contacts.update_period = self.sim.dt
        self.scene.left_fingertip_contacts.update_period = self.sim.dt
        self.scene.left_thumb_contacts.update_period = self.sim.dt
        self.scene.left_finger_pad_contacts.update_period = self.sim.dt
        self.scene.left_palm_contacts.update_period = self.sim.dt

        # Physics settings
        self.sim.physx.bounce_threshold_velocity = 0.01
        self.sim.physx.gpu_found_lost_aggregate_pairs_capacity = 1024 * 1024 * 4
        self.sim.physx.gpu_total_aggregate_pairs_capacity = 16 * 1024
        self.sim.physx.friction_correlation_distance = 0.00625

        # Event manager for manual resets
        self.event_manager = SimpleEventManager()

        self.event_manager.register("reset_object_self", SimpleEvent(
            func=lambda env: base_mdp.reset_root_state_uniform(
                env,
                torch.arange(env.num_envs, device=env.device),
                pose_range={"x": [-0.02, 0.02], "y": [-0.02, 0.02]},
                velocity_range={},
                asset_cfg=SceneEntityCfg("object"),
            )
        ))

        self.event_manager.register("reset_all_self", SimpleEvent(
            func=lambda env: base_mdp.reset_scene_to_default(
                env,
                torch.arange(env.num_envs, device=env.device)
            )
        ))
