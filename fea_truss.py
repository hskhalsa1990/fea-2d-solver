"""2D Truss Finite Element Solver (linear elastic)."""
from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Dict

@dataclass
class TrussElement:
    n1: int
    n2: int
    E: float          # Young's modulus
    A: float          # cross-section area

def element_stiffness_truss(node_coords: np.ndarray, elem: TrussElement) -> np.ndarray:
    """4x4 local stiffness in global coordinates."""
    x1, y1 = node_coords[elem.n1]
    x2, y2 = node_coords[elem.n2]
    L = np.hypot(x2 - x1, y2 - y1)
    if L < 1e-12:
        raise ValueError("Zero-length truss element")
    c = (x2 - x1) / L
    s = (y2 - y1) / L
    k = (elem.E * elem.A / L) * np.array([
        [ c*c,  c*s, -c*c, -c*s],
        [ c*s,  s*s, -c*s, -s*s],
        [-c*c, -c*s,  c*c,  c*s],
        [-c*s, -s*s,  c*s,  s*s],
    ])
    return k

def assemble_truss(node_coords: np.ndarray, elements: List[TrussElement]) -> np.ndarray:
    n = len(node_coords)
    K = np.zeros((2*n, 2*n))
    for e in elements:
        ke = element_stiffness_truss(node_coords, e)
        dofs = [2*e.n1, 2*e.n1+1, 2*e.n2, 2*e.n2+1]
        for i, di in enumerate(dofs):
            for j, dj in enumerate(dofs):
                K[di, dj] += ke[i, j]
    return K

def apply_bc_and_solve(K: np.ndarray, F: np.ndarray, fixed_dofs: List[int]) -> np.ndarray:
    """Reduce system and solve Ku = F."""
    n = K.shape[0]
    free = np.array([i for i in range(n) if i not in fixed_dofs])
    Kff = K[np.ix_(free, free)]
    Ff = F[free]
    uf = np.linalg.solve(Kff, Ff)
    u = np.zeros(n)
    u[free] = uf
    return u

def element_forces_truss(node_coords: np.ndarray, elements: List[TrussElement], u: np.ndarray) -> np.ndarray:
    """Axial force in each element (positive = tension)."""
    forces = []
    for e in elements:
        x1, y1 = node_coords[e.n1]
        x2, y2 = node_coords[e.n2]
        L = np.hypot(x2-x1, y2-y1)
        c = (x2-x1)/L
        s = (y2-y1)/L
        ue = u[[2*e.n1, 2*e.n1+1, 2*e.n2, 2*e.n2+1]]
        # local axial displacement difference
        delta = (-c*ue[0] - s*ue[1] + c*ue[2] + s*ue[3])
        forces.append(e.E * e.A / L * delta)
    return np.array(forces)

def solve_truss(node_coords: np.ndarray, elements: List[TrussElement],
                loads: Dict[int, Tuple[float, float]], fixed_nodes: List[int]) -> dict:
    """
    High-level truss solver.
    loads: {node_id: (Fx, Fy)}
    fixed_nodes: list of node ids that are fully fixed (ux=uy=0)
    """
    n = len(node_coords)
    K = assemble_truss(node_coords, elements)
    F = np.zeros(2*n)
    for nid, (fx, fy) in loads.items():
        F[2*nid] += fx
        F[2*nid+1] += fy
    fixed_dofs = []
    for nid in fixed_nodes:
        fixed_dofs.extend([2*nid, 2*nid+1])
    u = apply_bc_and_solve(K, F, fixed_dofs)
    forces = element_forces_truss(node_coords, elements, u)
    reactions = K @ u - F
    return {
        "displacements": u.reshape(-1, 2),
        "axial_forces": forces,
        "reactions": reactions.reshape(-1, 2),
        "max_disp": float(np.max(np.abs(u))),
        "max_force": float(np.max(np.abs(forces))),
    }
