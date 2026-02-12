# Copyright (c) 2025, Unitree Robotics Co., Ltd. All Rights Reserved.
# License: Apache License, Version 2.0
"""
TacSL Visuo-Tactile Sensor Configuration for Inspire Hand.

Provides sensor configuration presets for all 17 tactile regions per hand
(34 total sensors) matching the real RH56DFTP inspire hand sensor layout.

Sensor regions and their taxel counts:
- Fingertip (tip): 3x3 = 9 taxels per finger
- Finger nail (top): 12x8 = 96 taxels per finger
- Finger pad (palm): 10x8 = 80 taxels per finger (12x8 = 96 for thumb)
- Thumb middle: 3x3 = 9 taxels
- Palm: 8x14 = 112 taxels

Total per hand: 531 taxels (1,062 for both hands)

Usage:
    from tasks.common_config import InspireHandTactilePresets

    # Get individual sensor configs
    l_index_tip = InspireHandTactilePresets.get_finger_tip_sensor("L", "index")

    # Get all sensors for a hand
    left_sensors = InspireHandTactilePresets.get_all_hand_sensors("L")
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
from isaaclab.utils import configclass

# Try to import TacSL sensor config, fallback to None if not available
# In Isaac Lab v2.3.2+, TacSL is in isaaclab_contrib.sensors.tacsl_sensor
try:
    from isaaclab_contrib.sensors.tacsl_sensor import VisuoTactileSensorCfg
    TACSL_AVAILABLE = True
except ImportError:
    # Try alternate import paths
    try:
        from isaaclab_contrib.sensors import VisuoTactileSensorCfg
        TACSL_AVAILABLE = True
    except ImportError:
        VisuoTactileSensorCfg = None
        TACSL_AVAILABLE = False
        print("[tactile_configs] Warning: TacSL (isaaclab_contrib) not available. "
              "TacSL sensor configs will not be functional. "
              "Requires Isaac Lab v2.3.2+ with: pip install -e source/isaaclab_contrib")


@dataclass
class TactileRegionSpec:
    """Specification for a tactile sensor region."""
    grid_rows: int
    grid_cols: int
    link_suffix: str  # e.g., "intermediate", "proximal", "distal"
    idl_field_suffix: str  # e.g., "tip_touch", "top_touch", "palm_touch"

    @property
    def taxel_count(self) -> int:
        return self.grid_rows * self.grid_cols

    @property
    def grid_size(self) -> Tuple[int, int]:
        return (self.grid_rows, self.grid_cols)


# Define tactile region specifications
TACTILE_REGION_SPECS = {
    # Finger regions (index, middle, ring, pinky)
    "finger_tip": TactileRegionSpec(3, 3, "intermediate", "tip_touch"),
    "finger_nail": TactileRegionSpec(12, 8, "intermediate", "top_touch"),
    "finger_pad": TactileRegionSpec(10, 8, "proximal", "palm_touch"),

    # Thumb regions (different link structure)
    "thumb_tip": TactileRegionSpec(3, 3, "distal", "tip_touch"),
    "thumb_nail": TactileRegionSpec(12, 8, "distal", "top_touch"),
    "thumb_middle": TactileRegionSpec(3, 3, "intermediate", "middle_touch"),
    "thumb_pad": TactileRegionSpec(12, 8, "proximal", "palm_touch"),

    # Palm region
    "palm": TactileRegionSpec(8, 14, "palm_link", "palm_touch"),
}

# Finger name mapping: simulation prefix -> IDL field prefix
FINGER_TO_IDL = {
    "pinky": "fingerone",
    "ring": "fingertwo",
    "middle": "fingerthree",
    "index": "fingerfour",
    "thumb": "fingerfive",
}

# Ordered list of regular fingers (not thumb)
REGULAR_FINGERS = ["pinky", "ring", "middle", "index"]

# All fingers including thumb
ALL_FINGERS = ["pinky", "ring", "middle", "index", "thumb"]


@configclass
class TactileSensorBaseCfg:
    """Base configuration for TacSL tactile sensors.

    Provides common parameters for visuo-tactile sensor configuration.
    Matches the VisuoTactileSensorCfg API from Isaac Lab v2.3.2+.
    """

    # Default physics parameters for elastomer contact
    DEFAULT_NORMAL_STIFFNESS: float = 1.0
    DEFAULT_FRICTION: float = 2.0
    DEFAULT_TANGENTIAL_STIFFNESS: float = 0.1
    DEFAULT_UPDATE_PERIOD: float = 0.02  # 50Hz
    DEFAULT_TACTILE_MARGIN: float = 0.001  # 1mm margin from elastomer edges

    @classmethod
    def get_sensor_config(
        cls,
        prim_path: str,
        grid_size: Tuple[int, int],
        contact_object_expr: str = "{ENV_REGEX_NS}/Object.*",
        normal_stiffness: float = None,
        friction: float = None,
        tangential_stiffness: float = None,
        update_period: float = None,
        tactile_margin: float = None,
        enable_camera: bool = False,
        render_cfg: Any = None,
    ) -> Optional[Any]:
        """Create a TacSL VisuoTactileSensorCfg.

        Uses the official Isaac Lab v2.3.2+ VisuoTactileSensorCfg API.

        Args:
            prim_path: USD prim path to the elastomer geometry
            grid_size: (rows, cols) tuple for tactile array dimensions
            contact_object_expr: Prim path expression for contact objects
            normal_stiffness: Contact stiffness (N/m)
            friction: Friction coefficient
            tangential_stiffness: Shear stiffness
            update_period: Sensor update period in seconds
            tactile_margin: Margin from elastomer edges in meters
            enable_camera: Whether to enable GelSight camera output
            render_cfg: GelSightRenderCfg for camera rendering (required if enable_camera=True)

        Returns:
            VisuoTactileSensorCfg if TacSL available, None otherwise
        """
        if not TACSL_AVAILABLE:
            return None

        # Build config kwargs
        cfg_kwargs = {
            "prim_path": prim_path,
            "tactile_array_size": grid_size,
            "tactile_margin": tactile_margin or cls.DEFAULT_TACTILE_MARGIN,
            "enable_force_field": True,
            "enable_camera_tactile": enable_camera,
            "contact_object_prim_path_expr": contact_object_expr,
            "normal_contact_stiffness": normal_stiffness or cls.DEFAULT_NORMAL_STIFFNESS,
            "friction_coefficient": friction or cls.DEFAULT_FRICTION,
            "tangential_stiffness": tangential_stiffness or cls.DEFAULT_TANGENTIAL_STIFFNESS,
            "update_period": update_period or cls.DEFAULT_UPDATE_PERIOD,
        }

        # Add render_cfg if camera is enabled
        if enable_camera and render_cfg is not None:
            cfg_kwargs["render_cfg"] = render_cfg

        return VisuoTactileSensorCfg(**cfg_kwargs)


@configclass
class InspireHandTactilePresets:
    """TacSL sensor configuration presets for Inspire Hand (RH56DFTP).

    Provides factory methods to create sensor configurations for all
    tactile regions on the Inspire hand matching the real sensor layout.
    """

    @classmethod
    def is_available(cls) -> bool:
        """Check if TacSL sensors are available."""
        return TACSL_AVAILABLE

    @classmethod
    def get_finger_tip_sensor(
        cls,
        side: str,
        finger: str,
        contact_object_expr: str = "{ENV_REGEX_NS}/Object.*",
    ) -> Optional[Any]:
        """Get sensor config for finger tip region (3x3 = 9 taxels).

        Args:
            side: "L" or "R" for left/right hand
            finger: One of "pinky", "ring", "middle", "index", "thumb"
            contact_object_expr: Prim path expression for contact objects

        Returns:
            VisuoTactileSensorCfg or None if TacSL not available
        """
        if finger == "thumb":
            spec = TACTILE_REGION_SPECS["thumb_tip"]
        else:
            spec = TACTILE_REGION_SPECS["finger_tip"]

        prim_path = f"{{ENV_REGEX_NS}}/Robot/{side}_{finger}_{spec.link_suffix}/elastomer_tip"

        return TactileSensorBaseCfg.get_sensor_config(
            prim_path=prim_path,
            grid_size=spec.grid_size,
            contact_object_expr=contact_object_expr,
        )

    @classmethod
    def get_finger_nail_sensor(
        cls,
        side: str,
        finger: str,
        contact_object_expr: str = "{ENV_REGEX_NS}/Object.*",
    ) -> Optional[Any]:
        """Get sensor config for finger nail/top region (12x8 = 96 taxels).

        Args:
            side: "L" or "R" for left/right hand
            finger: One of "pinky", "ring", "middle", "index", "thumb"
            contact_object_expr: Prim path expression for contact objects

        Returns:
            VisuoTactileSensorCfg or None if TacSL not available
        """
        if finger == "thumb":
            spec = TACTILE_REGION_SPECS["thumb_nail"]
        else:
            spec = TACTILE_REGION_SPECS["finger_nail"]

        prim_path = f"{{ENV_REGEX_NS}}/Robot/{side}_{finger}_{spec.link_suffix}/elastomer_nail"

        return TactileSensorBaseCfg.get_sensor_config(
            prim_path=prim_path,
            grid_size=spec.grid_size,
            contact_object_expr=contact_object_expr,
        )

    @classmethod
    def get_finger_pad_sensor(
        cls,
        side: str,
        finger: str,
        contact_object_expr: str = "{ENV_REGEX_NS}/Object.*",
    ) -> Optional[Any]:
        """Get sensor config for finger pad region (10x8 or 12x8 taxels).

        Args:
            side: "L" or "R" for left/right hand
            finger: One of "pinky", "ring", "middle", "index", "thumb"
            contact_object_expr: Prim path expression for contact objects

        Returns:
            VisuoTactileSensorCfg or None if TacSL not available
        """
        if finger == "thumb":
            spec = TACTILE_REGION_SPECS["thumb_pad"]
        else:
            spec = TACTILE_REGION_SPECS["finger_pad"]

        prim_path = f"{{ENV_REGEX_NS}}/Robot/{side}_{finger}_{spec.link_suffix}/elastomer_pad"

        return TactileSensorBaseCfg.get_sensor_config(
            prim_path=prim_path,
            grid_size=spec.grid_size,
            contact_object_expr=contact_object_expr,
        )

    @classmethod
    def get_thumb_middle_sensor(
        cls,
        side: str,
        contact_object_expr: str = "{ENV_REGEX_NS}/Object.*",
    ) -> Optional[Any]:
        """Get sensor config for thumb middle region (3x3 = 9 taxels).

        Args:
            side: "L" or "R" for left/right hand
            contact_object_expr: Prim path expression for contact objects

        Returns:
            VisuoTactileSensorCfg or None if TacSL not available
        """
        spec = TACTILE_REGION_SPECS["thumb_middle"]
        prim_path = f"{{ENV_REGEX_NS}}/Robot/{side}_thumb_{spec.link_suffix}/elastomer_middle"

        return TactileSensorBaseCfg.get_sensor_config(
            prim_path=prim_path,
            grid_size=spec.grid_size,
            contact_object_expr=contact_object_expr,
        )

    @classmethod
    def get_palm_sensor(
        cls,
        side: str,
        contact_object_expr: str = "{ENV_REGEX_NS}/Object.*",
    ) -> Optional[Any]:
        """Get sensor config for palm region (8x14 = 112 taxels).

        Args:
            side: "L" or "R" for left/right hand
            contact_object_expr: Prim path expression for contact objects

        Returns:
            VisuoTactileSensorCfg or None if TacSL not available
        """
        spec = TACTILE_REGION_SPECS["palm"]
        # Palm link naming convention: left_palm_link or right_palm_link
        side_full = "left" if side == "L" else "right"
        prim_path = f"{{ENV_REGEX_NS}}/Robot/{side_full}_palm_link/elastomer_palm"

        return TactileSensorBaseCfg.get_sensor_config(
            prim_path=prim_path,
            grid_size=spec.grid_size,
            contact_object_expr=contact_object_expr,
        )

    @classmethod
    def get_all_finger_sensors(
        cls,
        side: str,
        finger: str,
        contact_object_expr: str = "{ENV_REGEX_NS}/Object.*",
    ) -> Dict[str, Optional[Any]]:
        """Get all sensor configs for a single finger.

        Args:
            side: "L" or "R" for left/right hand
            finger: One of "pinky", "ring", "middle", "index", "thumb"
            contact_object_expr: Prim path expression for contact objects

        Returns:
            Dict mapping sensor region names to configs
        """
        sensors = {
            f"{side}_{finger}_tip": cls.get_finger_tip_sensor(side, finger, contact_object_expr),
            f"{side}_{finger}_nail": cls.get_finger_nail_sensor(side, finger, contact_object_expr),
            f"{side}_{finger}_pad": cls.get_finger_pad_sensor(side, finger, contact_object_expr),
        }

        # Add thumb middle region
        if finger == "thumb":
            sensors[f"{side}_thumb_middle"] = cls.get_thumb_middle_sensor(side, contact_object_expr)

        return sensors

    @classmethod
    def get_all_hand_sensors(
        cls,
        side: str,
        contact_object_expr: str = "{ENV_REGEX_NS}/Object.*",
    ) -> Dict[str, Optional[Any]]:
        """Get all sensor configs for an entire hand (17 sensors).

        Args:
            side: "L" or "R" for left/right hand
            contact_object_expr: Prim path expression for contact objects

        Returns:
            Dict mapping sensor names to configs (17 sensors total)
        """
        sensors = {}

        # Add all finger sensors
        for finger in ALL_FINGERS:
            sensors.update(cls.get_all_finger_sensors(side, finger, contact_object_expr))

        # Add palm sensor
        sensors[f"{side}_palm"] = cls.get_palm_sensor(side, contact_object_expr)

        return sensors

    @classmethod
    def get_all_sensors(
        cls,
        contact_object_expr: str = "{ENV_REGEX_NS}/Object.*",
    ) -> Dict[str, Optional[Any]]:
        """Get all sensor configs for both hands (34 sensors total).

        Args:
            contact_object_expr: Prim path expression for contact objects

        Returns:
            Dict mapping sensor names to configs (34 sensors total)
        """
        sensors = {}
        sensors.update(cls.get_all_hand_sensors("L", contact_object_expr))
        sensors.update(cls.get_all_hand_sensors("R", contact_object_expr))
        return sensors

    @classmethod
    def get_sensor_idl_mapping(cls) -> Dict[str, str]:
        """Get mapping from sensor names to IDL field names.

        Returns:
            Dict mapping sensor names (e.g., "L_index_tip") to IDL fields
            (e.g., "fingerfour_tip_touch")
        """
        mapping = {}

        for side in ["L", "R"]:
            for finger in ALL_FINGERS:
                idl_prefix = FINGER_TO_IDL[finger]

                # Tip region
                mapping[f"{side}_{finger}_tip"] = f"{idl_prefix}_tip_touch"

                # Nail/top region
                mapping[f"{side}_{finger}_nail"] = f"{idl_prefix}_top_touch"

                # Pad/palm region
                mapping[f"{side}_{finger}_pad"] = f"{idl_prefix}_palm_touch"

                # Thumb middle region
                if finger == "thumb":
                    mapping[f"{side}_thumb_middle"] = f"{idl_prefix}_middle_touch"

            # Palm region
            mapping[f"{side}_palm"] = "palm_touch"

        return mapping

    @classmethod
    def get_total_taxels_per_hand(cls) -> int:
        """Get total number of taxels per hand.

        Returns:
            531 (total taxels per hand)
        """
        total = 0

        # Regular fingers: tip(9) + nail(96) + pad(80) = 185 each, 4 fingers = 740
        for finger in REGULAR_FINGERS:
            total += TACTILE_REGION_SPECS["finger_tip"].taxel_count
            total += TACTILE_REGION_SPECS["finger_nail"].taxel_count
            total += TACTILE_REGION_SPECS["finger_pad"].taxel_count

        # Thumb: tip(9) + nail(96) + middle(9) + pad(96) = 210
        total += TACTILE_REGION_SPECS["thumb_tip"].taxel_count
        total += TACTILE_REGION_SPECS["thumb_nail"].taxel_count
        total += TACTILE_REGION_SPECS["thumb_middle"].taxel_count
        total += TACTILE_REGION_SPECS["thumb_pad"].taxel_count

        # Palm: 112
        total += TACTILE_REGION_SPECS["palm"].taxel_count

        return total


# Export public API
__all__ = [
    "TACSL_AVAILABLE",
    "TactileRegionSpec",
    "TACTILE_REGION_SPECS",
    "FINGER_TO_IDL",
    "REGULAR_FINGERS",
    "ALL_FINGERS",
    "TactileSensorBaseCfg",
    "InspireHandTactilePresets",
]
