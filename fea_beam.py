"""Euler-Bernoulli 2D Beam Finite Element Solver."""
from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Tuple

@dataclass
class BeamElement:
    n1: int
    n2: int
    E: float
    I: float      # second moment of area
    A: float = 1.0  # optional axial

def element_stiffness_beam(node_coords: np.ndarray, elem: BeamElement) -> np.ndarray:
    """6x6 local stiffness (ux, uy, rz per node) transformed to global."""
    x1, y1 = node_coords[elem.n1]
    x2, y2 = node_coords[elem.n2]
    L = np.hypot(x2-x1, y2-y1)
    c = (x2-x1)/L
    s = (y2-y1)/L
    E, I, A = elem.E, elem.I, elem.A
    # local stiffness (order: u1, v1, r1, u2, v2, r2)
    k_local = np.array([
        [ A*E/L,  0,           0,          -A*E/L, 0,           0         ],
        [ 0,      12*E*I/L**3, 6*E*I/L**2,  0,    -12*E*I/L**3, 6*E*I/L**2],
        [ 0,      6*E*I/L**2,  4*E*I/L,     0,    -6*E*I/L**2,  2*E*I/L  ],
        [-A*E/L,  0,           0,           A*E/L, 0,           0         ],
        [ 0,     -12*E*I/L**3,-6*E*I/L**2,  0,     12*E*I/L**3,-6*E*I/L**2],
        [ 0,      6*E*I/L**2,  2*E*I/L,     0,    -6*E*I/L**2,  4*E*I/L  ],
    ])
    # transformation
    T = np.array([
        [c, s, 0, 0, 0, 0],
        [-s, c, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0],
        [0, 0, 0, c, s, 0],
        [0, 0, 0,-s, c, 0],
        [0, 0, 0, 0, 0, 1],
    ])
    return T.T @ k_local @ T

def assemble_beam(node_coords: np.ndarray, elements: List[BeamElement]) -> np.ndarray:
    n = len(node_coords)
    ndof = 3 * n
    K = np.zeros((ndof, ndof))
    for e in elements:
        ke = element_stiffness_beam(node_coords, e)
        dofs = [3*e.n1, 3*e.n1+1, 3*e.n1+2, 3*e.n2, 3*e.n2+1, 3*e.n2+2]
        for i, di in enumerate(dofs):
            for j, dj in enumerate(dofs):
                K[di, dj] += ke[i, j]
    return K

def solve_beam(node_coords: np.ndarray, elements: List[BeamElement],
               loads: Dict[int, Tuple[float, float, float]],
               fixed_dofs: List[int]) -> dict:
    """
    loads: {node: (Fx, Fy, Mz)}
    fixed_dofs: global dof indices that are constrained to 0
    """
    n = len(node_coords)
    K = assemble_beam(node_coords, elements)
    F = np.zeros(3*n)
    for nid, (fx, fy, mz) in loads.items():
        F[3*nid] += fx
        F[3*nid+1] += fy
        F[3*nid+2] += mz
    free = np.array([i for i in range(3*n) if i not in fixed_dofs])
    uf = np.linalg.solve(K[np.ix_(free, free)], F[free])
    u = np.zeros(3*n)
    u[free] = uf
    disp = u.reshape(-1, 3)   # ux, uy, rz
    return {
        "displacements": disp,
        "max_deflection": float(np.max(np.abs(disp[:, 1]))),
        "max_rotation": float(np.max(np.abs(disp[:, 2]))),
        "u_global": u,
    }

def cantilever_analytical(L: float, P: float, E: float, I: float) -> float:
    """Tip deflection of cantilever with end load P: PL^3 / (3EI)."""
    return P * L**3 / (3 * E * I)
