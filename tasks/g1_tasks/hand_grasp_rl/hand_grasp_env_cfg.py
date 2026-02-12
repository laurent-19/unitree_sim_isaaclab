# Copyright (c) 2025, Unitree Robotics Co., Ltd. All Rights Reserved.
# License: Apache License, Version 2.0

"""Environment config for G1 hand grasp RL task - Stage 2 of two-stage pipeline.

This environment trains a hand-only grasping policy:
- Robot arm is frozen at pre-grasp position
- Object is spawned near hand (within grasp range)
- Policy controls only hand joints (6 DOF)
- Contact sensors provide grasp feedback
"""

from __future__ import annotations

import isaaclab.sim as sim_utils
import isaaclab.envs.mdp as mdp
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import (
    EventTermCfg as EventTerm,
    ObservationGroupCfg as ObsGroup,
    ObservationTermCfg as ObsTerm,
    RewardTermCfg as RewTerm,
    TerminationTermCfg as DoneTerm,
    SceneEntityCfg,
)
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass

from tasks.common_config import G1RobotPresets
from . import grasp_mdp


# -----------------------------------------------------------------------------
# Constants
# -----------------------------------------------------------------------------

# Right hand proximal joints (6 DOF action space)
RIGHT_HAND_PROXIMAL_JOINTS = [
    "R_index_proximal_joint",
    "R_middle_proximal_joint",
    "R_ring_proximal_joint",
    "R_pinky_proximal_joint",
    "R_thumb_proximal_pitch_joint",
    "R_thumb_proximal_yaw_joint",
]

# Right arm joints (to lock at pre-grasp)
RIGHT_ARM_JOINTS = [
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
]

# All joints to lock (everything except right hand proximal)
LOCKED_JOINTS = [
    "waist_.*",
    "left_hip_.*", "left_knee_.*", "left_ankle_.*",
    "right_hip_.*", "right_knee_.*", "right_ankle_.*",
    "left_shoulder_.*", "left_elbow_.*", "left_wrist_.*",
    "right_shoulder_.*", "right_elbow_.*",  # Lock arm
    "R_.*_intermediate_joint",  # Lock finger intermediate joints (coupled)
    "R_thumb_intermediate_joint",
    "R_thumb_distal_joint",
]

EE_BODY_NAME = "R_thumb_proximal"  # Hand reference frame

# High friction for stable grasping
HIGH_FRICTION_MAT = sim_utils.RigidBodyMaterialCfg(
    static_friction=2.0,
    dynamic_friction=2.0,
    restitution=0.0,
)


# -----------------------------------------------------------------------------
# Scene Configuration
# -----------------------------------------------------------------------------

@configclass
class HandGraspSceneCfg(InteractiveSceneCfg):
    """Scene for hand grasp task.

    - G1 robot with arm at pre-grasp pose
    - Cylinder object positioned near hand
    - Contact sensors on fingertips
    """

    # Ground + lights
    ground = AssetBaseCfg(
        prim_path="/World/defaultGroundPlane",
        spawn=sim_utils.GroundPlaneCfg()
    )

    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DistantLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
    )

    dome_light = AssetBaseCfg(
        prim_path="/World/Light/Dome",
        spawn=sim_utils.DomeLightCfg(intensity=2500.0, color=(0.95, 0.95, 1.0)),
    )

    # Robot - G1 with Inspire hand, fixed base
    robot: ArticulationCfg = G1RobotPresets.g1_29dof_inspire_base_fix()

    # Position robot with arm at pre-grasp pose
    robot.init_state.pos = (0.0, 0.0, 0.8)
    robot.init_state.rot = (1.0, 0.0, 0.0, 0.0)

    # Pre-grasp joint positions for right arm (reaching forward)
    robot.init_state.joint_pos = {
        # Arm at pre-grasp position
        "right_shoulder_pitch_joint": 0.3,
        "right_shoulder_roll_joint": -0.2,
        "right_shoulder_yaw_joint": 0.0,
        "right_elbow_joint": 0.5,
        # Hand open
        "R_index_proximal_joint": 0.0,
        "R_middle_proximal_joint": 0.0,
        "R_ring_proximal_joint": 0.0,
        "R_pinky_proximal_joint": 0.0,
        "R_thumb_proximal_pitch_joint": 0.2,
        "R_thumb_proximal_yaw_joint": 0.8,
    }

    # Grasp cylinder - positioned near hand
    grasp_cylinder: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/GraspCylinder",
        spawn=sim_utils.CylinderCfg(
            radius=0.02,
            height=0.12,
            physics_material=HIGH_FRICTION_MAT,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                rigid_body_enabled=True,
                kinematic_enabled=False,
                disable_gravity=False,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.08),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.8, 0.2, 0.2)),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            # Will be placed near hand by event
            pos=(0.35, -0.15, 0.9),
            rot=(1.0, 0.0, 0.0, 0.0),
        ),
    )

    # Contact sensors on fingertips
    fingertip_contacts = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot",
        body_names=[
            "R_index_intermediate",
            "R_middle_intermediate",
            "R_ring_intermediate",
            "R_pinky_intermediate",
            "R_thumb_distal",
        ],
        history_length=3,
        filter_prim_paths_expr=["{ENV_REGEX_NS}/GraspCylinder"],
    )


# -----------------------------------------------------------------------------
# Actions
# -----------------------------------------------------------------------------

@configclass
class ActionsCfg:
    """Action space: 6 DOF hand proximal joints."""

    hand_pos = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=RIGHT_HAND_PROXIMAL_JOINTS,
        scale=1.0,
        use_default_offset=True,
    )


# -----------------------------------------------------------------------------
# Observations (~35 dim)
# -----------------------------------------------------------------------------

@configclass
class ObservationsCfg:

    @configclass
    class PolicyCfg(ObsGroup):
        """Policy observations for hand grasp."""

        # Hand joint state (6 + 6 = 12)
        hand_joint_pos = ObsTerm(func=grasp_mdp.hand_joint_pos)
        hand_joint_vel = ObsTerm(func=grasp_mdp.hand_joint_vel)

        # Contact forces (5 fingers x 3D = 15)
        contact_forces = ObsTerm(
            func=grasp_mdp.contact_forces,
            params={"sensor_name": "fingertip_contacts"},
        )

        # Object pose relative to hand (7)
        object_pose = ObsTerm(
            func=grasp_mdp.object_pose_relative_to_hand,
            params={
                "object_cfg": SceneEntityCfg("grasp_cylinder"),
                "ee_cfg": SceneEntityCfg("robot", body_names=[EE_BODY_NAME]),
            },
        )

        # Last action (6)
        last_action = ObsTerm(
            func=mdp.last_action,
            params={"action_name": "hand_pos"},
        )

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


# -----------------------------------------------------------------------------
# Rewards
# -----------------------------------------------------------------------------

@configclass
class RewardsCfg:
    """Reward terms for hand grasp training."""

    # Contact rewards
    finger_contact = RewTerm(
        func=grasp_mdp.finger_contact_reward,
        weight=0.5,
        params={
            "sensor_name": "fingertip_contacts",
            "threshold": 0.5,
        },
    )

    multi_finger = RewTerm(
        func=grasp_mdp.multi_finger_contact,
        weight=1.0,
        params={
            "sensor_name": "fingertip_contacts",
            "min_fingers": 3,
            "threshold": 0.5,
        },
    )

    # Lift reward
    lift_progress = RewTerm(
        func=grasp_mdp.object_lift_reward,
        weight=2.0,
        params={
            "object_cfg": SceneEntityCfg("grasp_cylinder"),
            "target_height": 0.05,
        },
    )

    # Success bonus
    grasp_success = RewTerm(
        func=grasp_mdp.grasp_hold_success,
        weight=5.0,
        params={
            "object_cfg": SceneEntityCfg("grasp_cylinder"),
            "lift_height": 0.05,
            "hold_steps": 30,
        },
    )

    # Penalties
    drop_penalty = RewTerm(
        func=grasp_mdp.object_dropped,
        weight=-3.0,
        params={
            "object_cfg": SceneEntityCfg("grasp_cylinder"),
            "min_height": 0.7,
        },
    )

    # Smoothness
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-0.01)


# -----------------------------------------------------------------------------
# Terminations
# -----------------------------------------------------------------------------

@configclass
class TerminationsCfg:
    """Termination conditions."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)

    object_dropped = DoneTerm(
        func=grasp_mdp.object_below_threshold,
        params={
            "object_cfg": SceneEntityCfg("grasp_cylinder"),
            "min_height": 0.6,
        },
    )

    grasp_success = DoneTerm(
        func=grasp_mdp.grasp_success_termination,
        params={
            "object_cfg": SceneEntityCfg("grasp_cylinder"),
            "lift_height": 0.1,
            "hold_steps": 50,
        },
    )


# -----------------------------------------------------------------------------
# Events
# -----------------------------------------------------------------------------

@configclass
class EventsCfg:
    """Event configuration."""

    reset_scene = EventTerm(func=mdp.reset_scene_to_default, mode="reset")

    # Lock arm joints at pre-grasp position
    lock_arm = EventTerm(
        func=grasp_mdp.hold_arm_at_pregrasp,
        mode="interval",
        interval_range_s=(0.0, 0.0),
        params={"asset_cfg": SceneEntityCfg("robot")},
    )

    # Lock other joints (legs, waist, etc)
    lock_other_joints = EventTerm(
        func=mdp.reset_joints_by_offset,
        mode="interval",
        interval_range_s=(0.0, 0.0),
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=LOCKED_JOINTS),
            "position_range": (0.0, 0.0),
            "velocity_range": (0.0, 0.0),
        },
    )

    # Place object near hand on reset
    place_object = EventTerm(
        func=grasp_mdp.place_object_near_hand,
        mode="reset",
        params={
            "object_cfg": SceneEntityCfg("grasp_cylinder"),
            "robot_cfg": SceneEntityCfg("robot"),
            "ee_body_name": EE_BODY_NAME,
            "offset_local": (0.08, 0.0, -0.02),
        },
    )

    # Reset tracking buffers
    reset_hold_counter = EventTerm(
        func=grasp_mdp.reset_hold_counter,
        mode="reset",
    )

    reset_initial_height = EventTerm(
        func=grasp_mdp.reset_initial_object_height,
        mode="reset",
        params={"object_cfg": SceneEntityCfg("grasp_cylinder")},
    )

    # Randomize object position slightly
    randomize_object_pos = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("grasp_cylinder"),
            "pose_range": {
                "x": [-0.02, 0.02],
                "y": [-0.02, 0.02],
                "z": [-0.01, 0.01],
            },
            "velocity_range": {},
        },
    )


# -----------------------------------------------------------------------------
# Environment Config
# -----------------------------------------------------------------------------

@configclass
class HandGraspEnvCfg(ManagerBasedRLEnvCfg):
    """Hand grasp RL environment - Stage 2 of two-stage pipeline."""

    scene: HandGraspSceneCfg = HandGraspSceneCfg(num_envs=512, env_spacing=2.5)
    events: EventsCfg = EventsCfg()
    actions: ActionsCfg = ActionsCfg()
    observations: ObservationsCfg = ObservationsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()

    # No commands needed for grasp task
    commands = None
    curriculum = None

    def __post_init__(self):
        # Physics
        self.sim.dt = 0.01
        self.decimation = 2
        self.sim.render_interval = self.decimation

        # Episode length (4 seconds for grasp)
        self.episode_length_s = 4.0

        # Contact sensor update
        self.scene.fingertip_contacts.update_period = self.sim.dt

        # Physics settings for stable contacts
        self.sim.physx.bounce_threshold_velocity = 0.01
        self.sim.physx.gpu_found_lost_aggregate_pairs_capacity = 1024 * 1024 * 4
        self.sim.physx.gpu_total_aggregate_pairs_capacity = 16 * 1024
        self.sim.physx.friction_correlation_distance = 0.00625

        # Viewer
        self.viewer.lookat = (0.35, -0.1, 0.9)
        self.viewer.eye = (0.8, 0.4, 1.2)


@configclass
class HandGraspEnvCfg_PLAY(HandGraspEnvCfg):
    """Play config with 1 environment."""

    scene: HandGraspSceneCfg = HandGraspSceneCfg(num_envs=1, env_spacing=2.5)

    def __post_init__(self):
        super().__post_init__()
        self.sim.render_interval = 1
