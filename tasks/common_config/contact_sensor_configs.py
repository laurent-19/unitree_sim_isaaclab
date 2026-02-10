# Copyright (c) 2025, Unitree Robotics Co., Ltd. All Rights Reserved.
# License: Apache License, Version 2.0
"""
Contact sensor configurations for dexterous hands.
Provides pre-configured contact sensors for Inspire hand finger tips.
Maps to FORCE_ACT register (1582) on real Inspire hand hardware.
"""

from isaaclab.sensors import ContactSensorCfg


class InspireHandContactSensorCfg:
    """Contact sensor configurations for Inspire dexterous hand.
    
    Provides contact force sensing for each finger tip, mapping to the 
    FORCE_ACT register (1582) on the real Inspire hand.
    
    Real hardware register mapping:
        - FORCE_ACT (1582): Actual force applied to each finger (6 short = 12 bytes)
        - Returns force values for: index, middle, ring, pinky, thumb, thumb_yaw
    
    Usage:
        Add these to your scene configuration:
        ```python
        @configclass
        class MySceneCfg(InteractiveSceneCfg):
            # ... other scene elements ...
            
            # Add contact sensors for inspire hand
            contact_forces = InspireHandContactSensorCfg.all_fingers()
            # Or for specific fingers:
            # left_index_contact = InspireHandContactSensorCfg.left_index()
        ```
    """
    
    @staticmethod
    def all_fingers(history_length: int = 3, debug_vis: bool = False) -> ContactSensorCfg:
        """Contact sensor for all robot bodies including fingers.
        
        This provides contact sensing for all finger tips on both hands.
        The force data is extracted per-finger in the observation function.
        
        Args:
            history_length: Number of frames to keep contact history
            debug_vis: Enable debug visualization of contacts
            
        Returns:
            ContactSensorCfg configured for all finger contacts
        """
        return ContactSensorCfg(
            prim_path="/World/envs/env_.*/Robot/.*",
            history_length=history_length,
            track_air_time=True,
            debug_vis=debug_vis,
            # Filter to only detect contacts with manipulable objects
            # Note: Uses "Object" with capital O to match scene prim paths
            filter_prim_paths_expr=["/World/envs/env_.*/Object.*"],
        )
    
    @staticmethod
    def left_hand_fingers(history_length: int = 3, debug_vis: bool = False) -> ContactSensorCfg:
        """Contact sensor specifically for left hand finger tips."""
        return ContactSensorCfg(
            prim_path="/World/envs/env_.*/Robot/L_*",
            history_length=history_length,
            track_air_time=True,
            debug_vis=debug_vis,
        )
    
    @staticmethod
    def right_hand_fingers(history_length: int = 3, debug_vis: bool = False) -> ContactSensorCfg:
        """Contact sensor specifically for right hand finger tips."""
        return ContactSensorCfg(
            prim_path="/World/envs/env_.*/Robot/R_*",
            history_length=history_length,
            track_air_time=True,
            debug_vis=debug_vis,
        )
    
    @staticmethod
    def left_index(history_length: int = 3) -> ContactSensorCfg:
        """Contact sensor for left index finger tip only."""
        return ContactSensorCfg(
            prim_path="/World/envs/env_.*/Robot/L_index_intermediate",
            history_length=history_length,
            track_air_time=False,
            debug_vis=False,
        )
    
    @staticmethod
    def left_thumb(history_length: int = 3) -> ContactSensorCfg:
        """Contact sensor for left thumb tip only."""
        return ContactSensorCfg(
            prim_path="/World/envs/env_.*/Robot/L_thumb_distal",
            history_length=history_length,
            track_air_time=False,
            debug_vis=False,
        )


# Finger body name mapping for reference
# These names correspond to the USD prim names in the Inspire hand model
INSPIRE_FINGER_BODY_NAMES = {
    "left_hand": {
        "index": "L_index_intermediate",
        "middle": "L_middle_intermediate", 
        "ring": "L_ring_intermediate",
        "pinky": "L_pinky_intermediate",
        "thumb": "L_thumb_distal",
        "palm": "left_wrist_roll_link",
    },
    "right_hand": {
        "index": "R_index_intermediate",
        "middle": "R_middle_intermediate",
        "ring": "R_ring_intermediate", 
        "pinky": "R_pinky_intermediate",
        "thumb": "R_thumb_distal",
        "palm": "right_wrist_roll_link",
    }
}
