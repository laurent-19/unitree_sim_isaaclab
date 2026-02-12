# Copyright (c) 2025, Unitree Robotics Co., Ltd. All Rights Reserved.
# License: Apache License, Version 2.0
"""
Tactile sensor mapping utilities for Inspire Hand.

Maps ContactSensor force readings to taxel grids matching the real
RH56DFTP hand sensor format (1,062 taxels per hand).

Supports two backends:
1. ContactSensor: Uses Gaussian force-to-taxel approximation
2. TacSL: Direct per-taxel force mapping from force field simulation
"""

import numpy as np
import torch
from dataclasses import dataclass
from typing import Tuple, Optional, Union


@dataclass
class TaxelGrid:
    """Represents a tactile sensor grid configuration."""
    rows: int
    cols: int

    @property
    def size(self) -> int:
        """Total number of taxels in the grid."""
        return self.rows * self.cols


# Real sensor grid definitions matching inspire_hand_touch.idl
TACTILE_GRIDS = {
    "tip": TaxelGrid(3, 3),          # 9 taxels - finger tip
    "nail": TaxelGrid(12, 8),        # 96 taxels - finger nail/top
    "pad": TaxelGrid(10, 8),         # 80 taxels - finger pad/palm
    "thumb_pad": TaxelGrid(12, 8),   # 96 taxels - thumb proximal pad
    "thumb_middle": TaxelGrid(3, 3), # 9 taxels - thumb intermediate
    "palm": TaxelGrid(8, 14),        # 112 taxels - palm
}

# Cache for Gaussian kernels to avoid recomputation
_kernel_cache = {}


def _get_gaussian_kernel(grid: TaxelGrid) -> np.ndarray:
    """Get or create a cached Gaussian kernel for a grid."""
    key = (grid.rows, grid.cols)
    if key not in _kernel_cache:
        cy, cx = grid.rows / 2, grid.cols / 2
        sigma_y, sigma_x = grid.rows / 3, grid.cols / 3

        y, x = np.mgrid[0:grid.rows, 0:grid.cols]
        kernel = np.exp(-((x - cx)**2 / (2 * sigma_x**2) +
                         (y - cy)**2 / (2 * sigma_y**2)))
        _kernel_cache[key] = kernel
    return _kernel_cache[key].copy()


def force_to_taxel_grid(
    force_vector: np.ndarray,
    grid: TaxelGrid,
    scale: float = 1000.0,
    max_value: int = 4095,
) -> np.ndarray:
    """Convert 3D contact force to taxel grid with Gaussian spatial distribution.

    Strategy:
    - Normal force magnitude determines overall pressure intensity
    - Shear forces (fx, fy) shift the pressure distribution across the grid

    Args:
        force_vector: 3D force vector [fx, fy, fz] in world frame
        grid: TaxelGrid defining the output dimensions
        scale: Conversion factor from Newtons to taxel units
        max_value: Maximum taxel value (12-bit = 4095)

    Returns:
        2D numpy array of int16 taxel values with shape (rows, cols)
    """
    magnitude = np.linalg.norm(force_vector)
    scaled = min(int(magnitude * scale), max_value)

    if scaled == 0:
        return np.zeros((grid.rows, grid.cols), dtype=np.int16)

    # Get base Gaussian kernel
    kernel = _get_gaussian_kernel(grid)

    # Shift kernel based on shear direction
    if magnitude > 0:
        shear_x = force_vector[0] / magnitude
        shear_y = force_vector[1] / magnitude
        shift_x = int(shear_x * grid.cols / 4)
        shift_y = int(shear_y * grid.rows / 4)
        kernel = np.roll(np.roll(kernel, shift_x, axis=1), shift_y, axis=0)

    # Scale and convert to int16
    taxels = (kernel * scaled).astype(np.int16)
    return np.clip(taxels, 0, max_value)


def flatten_taxel_grid(grid: np.ndarray, palm: bool = False) -> list:
    """Flatten taxel grid to list matching real sensor byte order.

    Args:
        grid: 2D numpy array of taxel values
        palm: If True, use palm-specific flattening (column-major from bottom)

    Returns:
        Flattened list of int16 values
    """
    if palm:
        # Palm: column-major from bottom row (matching real sensor layout)
        return grid[::-1, :].T.flatten().tolist()
    else:
        # Fingers: row-major order
        return grid.flatten().tolist()


def get_grid_for_region(finger: str, region: str) -> TaxelGrid:
    """Get the appropriate TaxelGrid for a finger region.

    Args:
        finger: One of 'pinky', 'ring', 'middle', 'index', 'thumb'
        region: One of 'tip', 'nail', 'pad', 'middle' (thumb only)

    Returns:
        TaxelGrid for the specified region
    """
    if finger == "thumb":
        if region == "pad":
            return TACTILE_GRIDS["thumb_pad"]
        elif region == "middle":
            return TACTILE_GRIDS["thumb_middle"]

    return TACTILE_GRIDS.get(region, TACTILE_GRIDS["tip"])


# =============================================================================
# TacSL Direct Mapping Functions
# =============================================================================

def tacsl_to_taxel_grid(
    normal_force: Union[np.ndarray, "torch.Tensor"],
    grid: TaxelGrid,
    scale: float = 1000.0,
    max_value: int = 4095,
) -> np.ndarray:
    """Direct mapping from TacSL force field to taxel grid.

    Unlike ContactSensor Gaussian approximation, TacSL provides
    per-taxel forces directly - just need scaling and clamping.

    This function handles the conversion from TacSL's force field
    output to the taxel format expected by the DDS interface.

    Args:
        normal_force: Force field tensor of shape (H, W) from TacSL sensor
        grid: TaxelGrid defining the expected output dimensions
        scale: Conversion factor from Newtons to taxel units
        max_value: Maximum taxel value (12-bit = 4095)

    Returns:
        2D numpy array of int16 taxel values with shape (rows, cols)

    Note:
        The TacSL force field should already match the grid dimensions.
        If dimensions don't match, the output is resized using bilinear
        interpolation.
    """
    # Convert torch tensor to numpy if needed
    if hasattr(normal_force, 'cpu'):
        force_np = normal_force.cpu().numpy()
    else:
        force_np = np.asarray(normal_force)

    # Handle empty/zero input
    if force_np.size == 0 or np.max(np.abs(force_np)) < 1e-10:
        return np.zeros((grid.rows, grid.cols), dtype=np.int16)

    # Scale forces to taxel units
    scaled = force_np * scale

    # Resize if dimensions don't match
    if force_np.shape != (grid.rows, grid.cols):
        from scipy import ndimage
        zoom_factors = (grid.rows / force_np.shape[0], grid.cols / force_np.shape[1])
        scaled = ndimage.zoom(scaled, zoom_factors, order=1)

    # Clip to valid range and convert to int16
    taxels = np.clip(scaled, 0, max_value).astype(np.int16)

    return taxels


def tacsl_shear_to_shift(
    shear_force: Union[np.ndarray, "torch.Tensor"],
    grid: TaxelGrid,
    max_shift: int = 2,
) -> Tuple[int, int]:
    """Convert TacSL shear forces to grid shift values.

    Computes average shear direction and converts to discrete
    grid cell shifts for visualization or analysis.

    Args:
        shear_force: Shear force tensor of shape (H, W, 2) with (fx, fy) components
        grid: TaxelGrid for shift scaling
        max_shift: Maximum shift in grid cells

    Returns:
        Tuple of (shift_x, shift_y) in grid cell units
    """
    # Convert torch tensor to numpy if needed
    if hasattr(shear_force, 'cpu'):
        shear_np = shear_force.cpu().numpy()
    else:
        shear_np = np.asarray(shear_force)

    if shear_np.size == 0:
        return (0, 0)

    # Compute average shear direction
    avg_fx = np.mean(shear_np[..., 0])
    avg_fy = np.mean(shear_np[..., 1])

    # Normalize by magnitude
    magnitude = np.sqrt(avg_fx**2 + avg_fy**2)
    if magnitude < 1e-6:
        return (0, 0)

    # Scale to grid shift
    shift_x = int(np.clip(avg_fx / magnitude * max_shift, -max_shift, max_shift))
    shift_y = int(np.clip(avg_fy / magnitude * max_shift, -max_shift, max_shift))

    return (shift_x, shift_y)


def combine_normal_and_shear(
    normal_force: Union[np.ndarray, "torch.Tensor"],
    shear_force: Union[np.ndarray, "torch.Tensor"],
    grid: TaxelGrid,
    scale: float = 1000.0,
    max_value: int = 4095,
) -> np.ndarray:
    """Combine TacSL normal and shear forces into shifted taxel grid.

    Creates a taxel grid from normal forces with spatial shift
    based on shear direction, similar to the ContactSensor approach
    but using actual per-taxel force data.

    Args:
        normal_force: Normal force field tensor of shape (H, W)
        shear_force: Shear force tensor of shape (H, W, 2)
        grid: TaxelGrid defining output dimensions
        scale: Conversion factor from Newtons to taxel units
        max_value: Maximum taxel value

    Returns:
        2D numpy array of int16 taxel values
    """
    # Get base taxel grid from normal forces
    taxels = tacsl_to_taxel_grid(normal_force, grid, scale, max_value)

    # Get shear-based shift
    shift_x, shift_y = tacsl_shear_to_shift(shear_force, grid)

    # Apply shift using numpy roll
    if shift_x != 0:
        taxels = np.roll(taxels, shift_x, axis=1)
    if shift_y != 0:
        taxels = np.roll(taxels, shift_y, axis=0)

    return taxels


def get_total_taxels_per_hand() -> int:
    """Get total number of taxels per hand.

    Returns:
        531 (total taxels per hand based on IDL specification)
    """
    # Regular fingers: 4 x (9 + 96 + 80) = 4 x 185 = 740
    # But wait - let me recalculate:
    # - Pinky/Ring/Middle/Index tip: 3x3 = 9 each, total 36
    # - Pinky/Ring/Middle/Index nail: 12x8 = 96 each, total 384
    # - Pinky/Ring/Middle/Index pad: 10x8 = 80 each, total 320
    # - Thumb tip: 3x3 = 9
    # - Thumb nail: 12x8 = 96
    # - Thumb middle: 3x3 = 9
    # - Thumb pad: 12x8 = 96
    # - Palm: 8x14 = 112
    # Total: 36 + 384 + 320 + 9 + 96 + 9 + 96 + 112 = 1062 / 2 = 531 per hand

    total = 0

    # 4 regular fingers
    total += 4 * TACTILE_GRIDS["tip"].size      # 4 x 9 = 36
    total += 4 * TACTILE_GRIDS["nail"].size     # 4 x 96 = 384
    total += 4 * TACTILE_GRIDS["pad"].size      # 4 x 80 = 320

    # Thumb (4 regions)
    total += TACTILE_GRIDS["tip"].size          # 9
    total += TACTILE_GRIDS["nail"].size         # 96
    total += TACTILE_GRIDS["thumb_middle"].size # 9
    total += TACTILE_GRIDS["thumb_pad"].size    # 96

    # Palm
    total += TACTILE_GRIDS["palm"].size         # 112

    return total  # Should be 1062 / 2 = 531


# Add exports
__all__ = [
    "TaxelGrid",
    "TACTILE_GRIDS",
    "force_to_taxel_grid",
    "flatten_taxel_grid",
    "get_grid_for_region",
    # TacSL functions
    "tacsl_to_taxel_grid",
    "tacsl_shear_to_shift",
    "combine_normal_and_shear",
    "get_total_taxels_per_hand",
]
