"""2D Plane-Stress FEA — CST triangles + Q4 bilinear quads.

Upgrades vs v1:
- Q4 (4-node bilinear) elements with 2x2 Gauss integration
- Nodal stress averaging (area-weighted)
- Plate-with-circular-hole mesh + Kirsch validation helper
- Structured rect mesh with optional mixed CST/Q4
- Convenience solvers and post-process utilities
"""
from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional

@dataclass
class Material:
    E: float = 210e9
    nu: float = 0.3
    thickness: float = 0.01

def D_matrix(mat: Material) -> np.ndarray:
    E, nu = mat.E, mat.nu
    return (E / (1 - nu**2)) * np.array([
        [1, nu, 0],
        [nu, 1, 0],
        [0, 0, (1-nu)/2],
    ])

def von_mises(sx: float, sy: float, txy: float) -> float:
    return float(np.sqrt(sx**2 - sx*sy + sy**2 + 3*txy**2))

# ------------------------------------------------------------------ CST
def triangle_area(coords: np.ndarray) -> float:
    x1,y1 = coords[0]; x2,y2 = coords[1]; x3,y3 = coords[2]
    return 0.5 * ((x2-x1)*(y3-y1) - (x3-x1)*(y2-y1))

def B_matrix_cst(coords: np.ndarray) -> Tuple[np.ndarray, float]:
    x1,y1 = coords[0]; x2,y2 = coords[1]; x3,y3 = coords[2]
    A = triangle_area(coords)
    if abs(A) < 1e-18:
        raise ValueError("Degenerate triangle")
    b1,b2,b3 = y2-y3, y3-y1, y1-y2
    c1,c2,c3 = x3-x2, x1-x3, x2-x1
    B = (1/(2*A)) * np.array([
        [b1, 0, b2, 0, b3, 0],
        [0, c1, 0, c2, 0, c3],
        [c1, b1, c2, b2, c3, b3],
    ])
    return B, abs(A)

def element_stiffness_cst(coords: np.ndarray, mat: Material) -> np.ndarray:
    B, A = B_matrix_cst(coords)
    D = D_matrix(mat)
    return mat.thickness * A * (B.T @ D @ B)

# ------------------------------------------------------------------ Q4
def shape_q4(xi: float, eta: float) -> Tuple[np.ndarray, np.ndarray]:
    """N (4,) and dN/d(xi,eta) (2,4) at natural coords."""
    N = 0.25 * np.array([
        (1-xi)*(1-eta), (1+xi)*(1-eta),
        (1+xi)*(1+eta), (1-xi)*(1+eta)
    ])
    dN = 0.25 * np.array([
        [-(1-eta), (1-eta), (1+eta), -(1+eta)],
        [-(1-xi), -(1+xi), (1+xi), (1-xi)],
    ])
    return N, dN

def element_stiffness_q4(coords: np.ndarray, mat: Material) -> np.ndarray:
    """4-node bilinear quad, 2x2 Gauss quadrature. coords ordered CCW."""
    D = D_matrix(mat)
    ke = np.zeros((8, 8))
    gp = [-1/np.sqrt(3), 1/np.sqrt(3)]
    for xi in gp:
        for eta in gp:
            _, dN = shape_q4(xi, eta)
            J = dN @ coords          # 2x2
            detJ = np.linalg.det(J)
            if abs(detJ) < 1e-18:
                raise ValueError("Singular Jacobian in Q4")
            dN_dx = np.linalg.inv(J) @ dN   # 2x4
            B = np.zeros((3, 8))
            for a in range(4):
                B[0, 2*a]   = dN_dx[0, a]
                B[1, 2*a+1] = dN_dx[1, a]
                B[2, 2*a]   = dN_dx[1, a]
                B[2, 2*a+1] = dN_dx[0, a]
            ke += mat.thickness * (B.T @ D @ B) * detJ
    return ke

def stress_at_q4_center(coords: np.ndarray, ue: np.ndarray, mat: Material) -> np.ndarray:
    """Stress [sx,sy,txy] at element center (xi=eta=0)."""
    D = D_matrix(mat)
    _, dN = shape_q4(0.0, 0.0)
    J = dN @ coords
    dN_dx = np.linalg.inv(J) @ dN
    B = np.zeros((3, 8))
    for a in range(4):
        B[0, 2*a] = dN_dx[0,a]; B[1, 2*a+1] = dN_dx[1,a]
        B[2, 2*a] = dN_dx[1,a]; B[2, 2*a+1] = dN_dx[0,a]
    return D @ (B @ ue)

# ------------------------------------------------------------------ Mesh
def generate_rect_mesh(Lx: float, Ly: float, nx: int, ny: int,
                       element: str = "CST") -> Tuple[np.ndarray, np.ndarray]:
    """Structured mesh. element='CST' → triangles, 'Q4' → quads."""
    xs = np.linspace(0, Lx, nx+1)
    ys = np.linspace(0, Ly, ny+1)
    nodes = np.array([[x, y] for y in ys for x in xs])
    def nid(i, j):
        return j*(nx+1) + i
    elements = []
    for j in range(ny):
        for i in range(nx):
            n0, n1 = nid(i,j), nid(i+1,j)
            n2, n3 = nid(i+1,j+1), nid(i,j+1)
            if element.upper() == "Q4":
                elements.append([n0, n1, n2, n3])
            else:
                elements.append([n0, n1, n2])
                elements.append([n0, n2, n3])
    return nodes, np.array(elements, dtype=int)

def generate_plate_with_hole(L: float = 1.0, R: float = 0.15, n_r: int = 6,
                             n_th: int = 16) -> Tuple[np.ndarray, np.ndarray]:
    """Quarter-plate with circular hole (symmetry). Polar-ish structured mesh → CST."""
    # radial lines from R to L (approx square outer)
    rs = np.linspace(R, L, n_r+1)
    thetas = np.linspace(0, np.pi/2, n_th+1)
    nodes = []
    for r in rs:
        for th in thetas:
            # blend to square outer boundary
            if r < L * 0.99:
                nodes.append([r*np.cos(th), r*np.sin(th)])
            else:
                # map to square side
                if th <= np.pi/4:
                    nodes.append([L, L*np.tan(th)])
                else:
                    nodes.append([L/np.tan(th) if th < np.pi/2 else 0, L])
    nodes = np.array(nodes)
    # fix last ring properly to square
    n_ang = n_th + 1
    for i, th in enumerate(thetas):
        idx = n_r * n_ang + i
        if th <= np.pi/4:
            nodes[idx] = [L, L * np.tan(th)]
        else:
            nodes[idx] = [L * np.tan(np.pi/2 - th), L]
    elements = []
    for ir in range(n_r):
        for it in range(n_th):
            n0 = ir*n_ang + it
            n1 = ir*n_ang + it + 1
            n2 = (ir+1)*n_ang + it + 1
            n3 = (ir+1)*n_ang + it
            elements.append([n0, n1, n2])
            elements.append([n0, n2, n3])
    return nodes, np.array(elements, dtype=int)

# ------------------------------------------------------------------ Assembly & solve
def assemble(nodes: np.ndarray, elements: np.ndarray, mat: Material) -> np.ndarray:
    n = len(nodes)
    K = np.zeros((2*n, 2*n))
    for el in elements:
        coords = nodes[el]
        if len(el) == 3:
            ke = element_stiffness_cst(coords, mat)
            nd = 3
        else:
            ke = element_stiffness_q4(coords, mat)
            nd = 4
        dofs = []
        for ni in el:
            dofs.extend([2*ni, 2*ni+1])
        for a, da in enumerate(dofs):
            for b, db in enumerate(dofs):
                K[da, db] += ke[a, b]
    return K

def apply_bc_solve(K: np.ndarray, F: np.ndarray, fixed_dofs: List[int]) -> np.ndarray:
    free = np.array([i for i in range(K.shape[0]) if i not in set(fixed_dofs)])
    uf = np.linalg.solve(K[np.ix_(free, free)], F[free])
    u = np.zeros(K.shape[0])
    u[free] = uf
    return u

def element_stresses(nodes: np.ndarray, elements: np.ndarray,
                     u: np.ndarray, mat: Material) -> np.ndarray:
    """(n_elem, 4) → sx, sy, txy, von_mises at element center."""
    D = D_matrix(mat)
    out = []
    for el in elements:
        coords = nodes[el]
        dofs = []
        for ni in el:
            dofs.extend([2*ni, 2*ni+1])
        ue = u[dofs]
        if len(el) == 3:
            B, _ = B_matrix_cst(coords)
            stress = D @ (B @ ue)
        else:
            stress = stress_at_q4_center(coords, ue, mat)
        out.append([stress[0], stress[1], stress[2], von_mises(*stress)])
    return np.array(out)

def nodal_stress_average(nodes: np.ndarray, elements: np.ndarray,
                         elem_stress: np.ndarray) -> np.ndarray:
    """Area-weighted average of element stresses onto nodes. Returns (n_nodes, 4)."""
    n = len(nodes)
    acc = np.zeros((n, 4))
    wsum = np.zeros(n)
    for eidx, el in enumerate(elements):
        coords = nodes[el]
        if len(el) == 3:
            A = abs(triangle_area(coords))
        else:
            # approximate area via two triangles
            A = abs(triangle_area(coords[[0,1,2]])) + abs(triangle_area(coords[[0,2,3]]))
        for ni in el:
            acc[ni] += A * elem_stress[eidx]
            wsum[ni] += A
    wsum = np.maximum(wsum, 1e-30)
    return acc / wsum[:, None]

def solve_plate(nodes: np.ndarray, elements: np.ndarray, mat: Material,
                loads: Dict[int, Tuple[float, float]],
                fixed_dofs: List[int]) -> dict:
    K = assemble(nodes, elements, mat)
    F = np.zeros(2*len(nodes))
    for nid, (fx, fy) in loads.items():
        F[2*nid] += fx
        F[2*nid+1] += fy
    u = apply_bc_solve(K, F, fixed_dofs)
    stresses = element_stresses(nodes, elements, u, mat)
    nodal = nodal_stress_average(nodes, elements, stresses)
    disp = u.reshape(-1, 2)
    return {
        "displacements": disp,
        "stresses": stresses,
        "nodal_stresses": nodal,
        "max_vm": float(np.max(stresses[:, 3])),
        "max_disp": float(np.max(np.linalg.norm(disp, axis=1))),
        "u_global": u,
        "nodes": nodes,
        "elements": elements,
        "mat": mat,
    }

def cantilever_plate_mesh(L: float = 1.0, H: float = 0.2, nx: int = 20, ny: int = 4,
                          E: float = 210e9, nu: float = 0.3, t: float = 0.01,
                          tip_load: float = -1000.0, element: str = "CST") -> dict:
    mat = Material(E=E, nu=nu, thickness=t)
    nodes, elements = generate_rect_mesh(L, H, nx, ny, element=element)
    fixed = []
    for i, (x, y) in enumerate(nodes):
        if abs(x) < 1e-9:
            fixed.extend([2*i, 2*i+1])
    right = [i for i, (x, y) in enumerate(nodes) if abs(x - L) < 1e-9]
    load_per = tip_load / max(len(right), 1)
    loads = {i: (0.0, load_per) for i in right}
    return solve_plate(nodes, elements, mat, loads, fixed)

def plate_with_hole_tension(L: float = 1.0, R: float = 0.15, n_r: int = 8, n_th: int = 20,
                            sigma_far: float = 100e6, E: float = 210e9, nu: float = 0.3,
                            t: float = 0.01) -> dict:
    """Quarter-plate with hole under remote uniaxial tension (Kirsch problem)."""
    mat = Material(E=E, nu=nu, thickness=t)
    nodes, elements = generate_plate_with_hole(L, R, n_r, n_th)
    fixed = []
    loads = {}
    for i, (x, y) in enumerate(nodes):
        # symmetry on x=0 (uy free, ux=0) and y=0 (ux free, uy=0)
        if abs(x) < 1e-8:
            fixed.append(2*i)       # ux = 0
        if abs(y) < 1e-8:
            fixed.append(2*i+1)     # uy = 0
        # traction on right edge x≈L
        if abs(x - L) < 1e-6:
            # approximate nodal force from far-field stress
            # simple: distribute sigma * t * dy
            loads[i] = loads.get(i, (0.0, 0.0))
    # better traction: find right-edge nodes and assign based on spacing
    right = sorted([i for i, (x, y) in enumerate(nodes) if abs(x - L) < 1e-5],
                   key=lambda i: nodes[i, 1])
    if len(right) >= 2:
        for k, i in enumerate(right):
            if k == 0:
                dy = 0.5 * (nodes[right[1], 1] - nodes[right[0], 1])
            elif k == len(right)-1:
                dy = 0.5 * (nodes[right[-1], 1] - nodes[right[-2], 1])
            else:
                dy = 0.5 * (nodes[right[k+1], 1] - nodes[right[k-1], 1])
            fx = sigma_far * t * dy
            loads[i] = (fx, 0.0)
    res = solve_plate(nodes, elements, mat, loads, fixed)
    # theoretical Kt ≈ 3 for circular hole in infinite plate
    res["Kt_theory"] = 3.0
    res["sigma_far"] = sigma_far
    return res

# keep old name for compatibility
def compute_element_stresses(*args, **kwargs):
    return element_stresses(*args, **kwargs)
