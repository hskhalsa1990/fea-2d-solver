"""Mesh helpers and simple structured generators."""
from __future__ import annotations
import numpy as np
from typing import Tuple

def rect_mesh(Lx: float, Ly: float, nx: int, ny: int) -> Tuple[np.ndarray, np.ndarray]:
    from fea_plate import generate_rect_mesh
    return generate_rect_mesh(Lx, Ly, nx, ny)

def plot_mesh_coords(nodes: np.ndarray, elements: np.ndarray):
    """Return lists of line segments for quick plotting."""
    segs = []
    for el in elements:
        for a, b in zip(el, np.roll(el, -1)):
            segs.append([nodes[a], nodes[b]])
    return segs
