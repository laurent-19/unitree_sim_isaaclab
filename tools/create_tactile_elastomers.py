#!/usr/bin/env python3
# Copyright (c) 2025, Unitree Robotics Co., Ltd. All Rights Reserved.
# License: Apache License, Version 2.0
"""
Create Tactile Elastomer Geometry for Inspire Hand USD.

This script adds elastomer geometry meshes to the Inspire Hand USD file
for use with TacSL visuo-tactile sensors. Each tactile region gets an
elastomer mesh positioned on the finger/palm surface.

Usage:
    # From Isaac Sim Python environment
    ./python.sh tools/create_tactile_elastomers.py --input <input.usd> --output <output.usd>

    # Or with docker
    ./run_docker.sh python tools/create_tactile_elastomers.py --input <input.usd> --output <output.usd>

The script creates:
- Soft body elastomer meshes at each tactile region
- Compliant contact physics materials
- TacSL sensor attachment points

Requires USD/Isaac Sim Python environment with pxr modules available.
"""

from __future__ import annotations

import argparse
import os
import sys
import math
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

# USD imports (requires Isaac Sim environment)
# Try to initialize Isaac Sim first if pxr is not directly available
USD_AVAILABLE = False
_simulation_app = None

def _init_isaac_sim():
    """Initialize Isaac Sim to get access to USD modules."""
    global _simulation_app
    if _simulation_app is not None:
        return True
    try:
        from isaacsim import SimulationApp
        _simulation_app = SimulationApp({"headless": True})
        return True
    except ImportError:
        return False

try:
    from pxr import Usd, UsdGeom, UsdPhysics, UsdShade, Gf, Sdf
    USD_AVAILABLE = True
except ImportError:
    # Try initializing Isaac Sim to get pxr access
    if _init_isaac_sim():
        try:
            from pxr import Usd, UsdGeom, UsdPhysics, UsdShade, Gf, Sdf
            USD_AVAILABLE = True
        except ImportError:
            print("Warning: USD (pxr) modules not available even after Isaac Sim init.")
    else:
        print("Warning: USD (pxr) modules not available. Run from Isaac Sim Python environment.")


@dataclass
class ElastomerSpec:
    """Specification for an elastomer mesh."""
    name: str
    grid_rows: int
    grid_cols: int
    width: float  # meters
    height: float  # meters
    thickness: float = 0.002  # 2mm default elastomer thickness
    offset: Tuple[float, float, float] = (0.0, 0.0, 0.0)  # position offset from link
    rotation: Tuple[float, float, float] = (0.0, 0.0, 0.0)  # euler rotation (degrees)


# Elastomer specifications for each tactile region
# Dimensions are approximate based on real Inspire hand sensor layout
ELASTOMER_SPECS = {
    # Finger regions (index, middle, ring, pinky)
    "finger_tip": ElastomerSpec(
        name="elastomer_tip",
        grid_rows=3, grid_cols=3,
        width=0.012, height=0.012,
        offset=(0.0, 0.0, 0.015),  # At fingertip
    ),
    "finger_nail": ElastomerSpec(
        name="elastomer_nail",
        grid_rows=12, grid_cols=8,
        width=0.024, height=0.016,
        offset=(0.0, 0.005, 0.01),  # On nail side
        rotation=(0.0, -90.0, 0.0),
    ),
    "finger_pad": ElastomerSpec(
        name="elastomer_pad",
        grid_rows=10, grid_cols=8,
        width=0.020, height=0.016,
        offset=(0.0, -0.005, 0.0),  # On pad/palm side
        rotation=(0.0, 90.0, 0.0),
    ),

    # Thumb regions (different geometry)
    "thumb_tip": ElastomerSpec(
        name="elastomer_tip",
        grid_rows=3, grid_cols=3,
        width=0.014, height=0.014,
        offset=(0.0, 0.0, 0.018),
    ),
    "thumb_nail": ElastomerSpec(
        name="elastomer_nail",
        grid_rows=12, grid_cols=8,
        width=0.028, height=0.018,
        offset=(0.0, 0.006, 0.012),
        rotation=(0.0, -90.0, 0.0),
    ),
    "thumb_middle": ElastomerSpec(
        name="elastomer_middle",
        grid_rows=3, grid_cols=3,
        width=0.012, height=0.012,
        offset=(0.0, 0.0, 0.01),
    ),
    "thumb_pad": ElastomerSpec(
        name="elastomer_pad",
        grid_rows=12, grid_cols=8,
        width=0.028, height=0.018,
        offset=(0.0, -0.006, 0.0),
        rotation=(0.0, 90.0, 0.0),
    ),

    # Palm region
    "palm": ElastomerSpec(
        name="elastomer_palm",
        grid_rows=8, grid_cols=14,
        width=0.06, height=0.04,
        offset=(0.0, 0.0, 0.01),
    ),
}

# Link paths for each tactile region
# {side}_{finger}_{link_type}
LINK_MAPPING = {
    # Regular fingers use intermediate for tip/nail, proximal for pad
    "finger": {
        "tip": "intermediate",
        "nail": "intermediate",
        "pad": "proximal",
    },
    # Thumb uses distal for tip/nail, intermediate for middle, proximal for pad
    "thumb": {
        "tip": "distal",
        "nail": "distal",
        "middle": "intermediate",
        "pad": "proximal",
    },
}

# List of regular fingers
FINGERS = ["index", "middle", "ring", "pinky"]


def create_elastomer_mesh(
    stage: Usd.Stage,
    parent_path: str,
    spec: ElastomerSpec,
) -> Optional[UsdGeom.Mesh]:
    """Create an elastomer mesh primitive.

    Args:
        stage: USD stage
        parent_path: Path to parent link prim
        spec: Elastomer specification

    Returns:
        UsdGeom.Mesh prim or None if failed
    """
    if not USD_AVAILABLE:
        return None

    mesh_path = f"{parent_path}/{spec.name}"

    # Create mesh prim
    mesh = UsdGeom.Mesh.Define(stage, mesh_path)

    # Generate simple quad grid for elastomer surface
    # This creates a flat grid that TacSL can use for force field sensing
    points, face_counts, face_indices = generate_grid_mesh(
        spec.grid_cols, spec.grid_rows,
        spec.width, spec.height,
        spec.thickness,
    )

    mesh.GetPointsAttr().Set(points)
    mesh.GetFaceVertexCountsAttr().Set(face_counts)
    mesh.GetFaceVertexIndicesAttr().Set(face_indices)

    # Set transform
    xform = UsdGeom.Xformable(mesh)

    # Apply translation
    translate_op = xform.AddTranslateOp()
    translate_op.Set(Gf.Vec3d(*spec.offset))

    # Apply rotation if specified
    if any(r != 0 for r in spec.rotation):
        rotate_op = xform.AddRotateXYZOp()
        rotate_op.Set(Gf.Vec3f(*spec.rotation))

    return mesh


def generate_grid_mesh(
    cols: int,
    rows: int,
    width: float,
    height: float,
    thickness: float,
) -> Tuple[List[Gf.Vec3f], List[int], List[int]]:
    """Generate a grid mesh for the elastomer surface.

    Creates a simple box-like mesh representing the elastomer pad.

    Args:
        cols: Number of grid columns
        rows: Number of grid rows
        width: Total width in meters
        height: Total height in meters
        thickness: Elastomer thickness in meters

    Returns:
        Tuple of (points, face_vertex_counts, face_vertex_indices)
    """
    points = []
    face_counts = []
    face_indices = []

    # Create top and bottom surfaces
    half_w = width / 2
    half_h = height / 2

    # Generate grid vertices for top surface
    for j in range(rows + 1):
        for i in range(cols + 1):
            x = -half_w + (i / cols) * width
            y = -half_h + (j / rows) * height
            # Top surface
            points.append(Gf.Vec3f(x, y, thickness))

    # Generate vertices for bottom surface
    for j in range(rows + 1):
        for i in range(cols + 1):
            x = -half_w + (i / cols) * width
            y = -half_h + (j / rows) * height
            # Bottom surface
            points.append(Gf.Vec3f(x, y, 0.0))

    num_top = (rows + 1) * (cols + 1)

    # Generate faces for top surface (facing +Z)
    for j in range(rows):
        for i in range(cols):
            v0 = j * (cols + 1) + i
            v1 = v0 + 1
            v2 = v0 + (cols + 1) + 1
            v3 = v0 + (cols + 1)
            face_counts.append(4)
            face_indices.extend([v0, v1, v2, v3])

    # Generate faces for bottom surface (facing -Z)
    for j in range(rows):
        for i in range(cols):
            v0 = num_top + j * (cols + 1) + i
            v1 = v0 + 1
            v2 = v0 + (cols + 1) + 1
            v3 = v0 + (cols + 1)
            face_counts.append(4)
            face_indices.extend([v3, v2, v1, v0])  # Reversed winding

    # Generate side faces
    # Left edge (X = -half_w)
    for j in range(rows):
        v0 = j * (cols + 1)
        v1 = (j + 1) * (cols + 1)
        v2 = num_top + (j + 1) * (cols + 1)
        v3 = num_top + j * (cols + 1)
        face_counts.append(4)
        face_indices.extend([v0, v3, v2, v1])

    # Right edge (X = +half_w)
    for j in range(rows):
        v0 = j * (cols + 1) + cols
        v1 = (j + 1) * (cols + 1) + cols
        v2 = num_top + (j + 1) * (cols + 1) + cols
        v3 = num_top + j * (cols + 1) + cols
        face_counts.append(4)
        face_indices.extend([v0, v1, v2, v3])

    # Bottom edge (Y = -half_h)
    for i in range(cols):
        v0 = i
        v1 = i + 1
        v2 = num_top + i + 1
        v3 = num_top + i
        face_counts.append(4)
        face_indices.extend([v0, v1, v2, v3])

    # Top edge (Y = +half_h)
    for i in range(cols):
        v0 = rows * (cols + 1) + i
        v1 = v0 + 1
        v2 = num_top + rows * (cols + 1) + i + 1
        v3 = num_top + rows * (cols + 1) + i
        face_counts.append(4)
        face_indices.extend([v0, v3, v2, v1])

    return points, face_counts, face_indices


def add_physics_material(
    stage: Usd.Stage,
    mesh_path: str,
    stiffness: float = 1000.0,
    damping: float = 100.0,
    friction: float = 2.0,
) -> None:
    """Add compliant contact physics material to elastomer.

    Args:
        stage: USD stage
        mesh_path: Path to the mesh prim
        stiffness: Contact stiffness (N/m)
        damping: Contact damping (N*s/m)
        friction: Friction coefficient
    """
    if not USD_AVAILABLE:
        return

    # Create physics material
    material_path = f"{mesh_path}/physics_material"

    # Add rigid body material
    material = UsdShade.Material.Define(stage, material_path)

    # Create physics material API
    prim = stage.GetPrimAtPath(mesh_path)
    if not prim:
        return

    # Apply collision API
    UsdPhysics.CollisionAPI.Apply(prim)

    # Create material with friction properties
    phys_mat_path = f"{mesh_path}/PhysicsMaterial"
    phys_mat = UsdPhysics.MaterialAPI.Apply(stage.DefinePrim(phys_mat_path))

    # Set material properties
    phys_mat.CreateStaticFrictionAttr(friction)
    phys_mat.CreateDynamicFrictionAttr(friction * 0.8)
    phys_mat.CreateRestitutionAttr(0.0)

    # Bind material to mesh
    mesh_prim = stage.GetPrimAtPath(mesh_path)
    if mesh_prim:
        binding_api = UsdShade.MaterialBindingAPI.Apply(mesh_prim)
        binding_api.Bind(material)


def add_elastomers_to_hand(
    stage: Usd.Stage,
    robot_path: str,
    side: str,
) -> List[str]:
    """Add all elastomer meshes to a hand.

    Args:
        stage: USD stage
        robot_path: Path to robot root prim
        side: "L" or "R" for left/right hand

    Returns:
        List of created elastomer prim paths
    """
    created = []

    # Add elastomers to regular fingers
    for finger in FINGERS:
        for region in ["tip", "nail", "pad"]:
            link_suffix = LINK_MAPPING["finger"][region]
            link_path = f"{robot_path}/{side}_{finger}_{link_suffix}"

            # Check if link exists
            if not stage.GetPrimAtPath(link_path):
                print(f"Warning: Link not found: {link_path}")
                continue

            spec = ELASTOMER_SPECS[f"finger_{region}"]
            mesh = create_elastomer_mesh(stage, link_path, spec)
            if mesh:
                mesh_path = mesh.GetPath().pathString
                add_physics_material(stage, mesh_path)
                created.append(mesh_path)
                print(f"Created: {mesh_path}")

    # Add elastomers to thumb
    for region in ["tip", "nail", "middle", "pad"]:
        link_suffix = LINK_MAPPING["thumb"][region]
        link_path = f"{robot_path}/{side}_thumb_{link_suffix}"

        if not stage.GetPrimAtPath(link_path):
            print(f"Warning: Link not found: {link_path}")
            continue

        spec = ELASTOMER_SPECS[f"thumb_{region}"]
        mesh = create_elastomer_mesh(stage, link_path, spec)
        if mesh:
            mesh_path = mesh.GetPath().pathString
            add_physics_material(stage, mesh_path)
            created.append(mesh_path)
            print(f"Created: {mesh_path}")

    # Add palm elastomer
    # Palm link naming: left_palm_link or right_palm_link
    side_full = "left" if side == "L" else "right"
    palm_path = f"{robot_path}/{side_full}_palm_link"

    # Fallback: try using the hand base link if palm_link doesn't exist
    if not stage.GetPrimAtPath(palm_path):
        # Try alternative paths
        alt_paths = [
            f"{robot_path}/{side_full}_hand_base_link",
            f"{robot_path}/{side}_thumb_proximal",  # Use thumb proximal as proxy
        ]
        for alt in alt_paths:
            if stage.GetPrimAtPath(alt):
                palm_path = alt
                break

    if stage.GetPrimAtPath(palm_path):
        spec = ELASTOMER_SPECS["palm"]
        mesh = create_elastomer_mesh(stage, palm_path, spec)
        if mesh:
            mesh_path = mesh.GetPath().pathString
            add_physics_material(stage, mesh_path)
            created.append(mesh_path)
            print(f"Created: {mesh_path}")
    else:
        print(f"Warning: Palm link not found for {side} hand")

    return created


def process_usd(input_path: str, output_path: str) -> bool:
    """Process USD file and add elastomers.

    Args:
        input_path: Path to input USD file
        output_path: Path to output USD file

    Returns:
        True if successful
    """
    if not USD_AVAILABLE:
        print("Error: USD modules not available")
        return False

    if not os.path.exists(input_path):
        print(f"Error: Input file not found: {input_path}")
        return False

    print(f"Loading: {input_path}")
    stage = Usd.Stage.Open(input_path)
    if not stage:
        print("Error: Failed to open USD stage")
        return False

    # Find robot root (usually /Robot or similar)
    robot_path = None
    for prim in stage.Traverse():
        prim_path = prim.GetPath().pathString
        # Look for inspire hand finger links to identify robot root
        if "L_index_proximal" in prim_path or "R_index_proximal" in prim_path:
            # Extract robot root path
            parts = prim_path.split("/")
            for i, part in enumerate(parts):
                if "index" in part.lower():
                    robot_path = "/".join(parts[:i])
                    break
            break

    if not robot_path:
        print("Error: Could not find Inspire hand links in USD")
        return False

    print(f"Robot root: {robot_path}")

    # Add elastomers to both hands
    created = []
    for side in ["L", "R"]:
        created.extend(add_elastomers_to_hand(stage, robot_path, side))

    print(f"\nCreated {len(created)} elastomer meshes")

    # Save output
    print(f"Saving: {output_path}")
    stage.Export(output_path)

    return True


def main():
    parser = argparse.ArgumentParser(
        description="Add tactile elastomer geometry to Inspire Hand USD"
    )
    parser.add_argument(
        "--input", "-i",
        required=True,
        help="Input USD file path"
    )
    parser.add_argument(
        "--output", "-o",
        required=True,
        help="Output USD file path"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be done without modifying files"
    )

    args = parser.parse_args()

    if args.dry_run:
        print("Dry run mode - no files will be modified")
        print(f"Would process: {args.input}")
        print(f"Would output to: {args.output}")

        print("\nElastomer regions that would be created:")
        for name, spec in ELASTOMER_SPECS.items():
            print(f"  {name}: {spec.grid_rows}x{spec.grid_cols} = {spec.grid_rows * spec.grid_cols} taxels")

        print(f"\nTotal sensors per hand: 17")
        print(f"Total sensors for both hands: 34")
        return

    success = process_usd(args.input, args.output)

    # Cleanup Isaac Sim if it was initialized
    if _simulation_app is not None:
        _simulation_app.close()

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
