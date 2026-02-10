# Copyright (c) 2025, Unitree Robotics Co., Ltd. All Rights Reserved.
# License: Apache License, Version 2.0
"""
Tactile sensor mapping utilities for Inspire Hand.

Maps ContactSensor force readings to taxel grids matching the real
RH56DFTP hand sensor format (1,062 taxels per hand).
"""

import numpy as np
from dataclasses import dataclass
from typing import Tuple


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
