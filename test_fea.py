"""Pytest suite for upgraded 2D FEA solvers."""
import numpy as np
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[0]))

from fea_truss import TrussElement, assemble_truss, apply_bc_and_solve, element_forces_truss
from fea_beam import BeamElement, solve_beam, cantilever_analytical
from fea_plate import (Material, generate_rect_mesh, solve_plate, cantilever_plate_mesh,
                       plate_with_hole_tension, von_mises, element_stiffness_q4,
                       nodal_stress_average)

def test_truss_simple_bar():
    nodes = np.array([[0.,0.], [1.,0.]])
    elems = [TrussElement(0, 1, E=210e9, A=0.001)]
    K = assemble_truss(nodes, elems)
    F = np.zeros(4); F[2] = 1000.0
    u = apply_bc_and_solve(K, F, [0, 1, 3])
    expected = 1000 * 1.0 / (210e9 * 0.001)
    assert abs(u[2] - expected) < 1e-10
    assert abs(element_forces_truss(nodes, elems, u)[0] - 1000) < 1e-4

def test_beam_cantilever_exact():
    L, E, I, P = 1.0, 210e9, 1e-6, -1000.0
    nodes = np.array([[0.,0.], [L,0.]])
    elems = [BeamElement(0, 1, E=E, I=I, A=0.01)]
    res = solve_beam(nodes, elems, loads={1: (0., P, 0.)}, fixed_dofs=[0,1,2])
    u_exact = cantilever_analytical(L, abs(P), E, I) * np.sign(P)
    assert abs(res["displacements"][1,1] - u_exact) / abs(u_exact) < 1e-6

def test_plate_cst_runs():
    res = cantilever_plate_mesh(L=1.0, H=0.2, nx=8, ny=2, tip_load=-500, element="CST")
    assert res["max_vm"] > 0 and res["max_disp"] > 0
    assert res["stresses"].shape[1] == 4
    assert res["nodal_stresses"].shape[0] == len(res["nodes"])

def test_plate_q4_runs():
    res = cantilever_plate_mesh(L=1.0, H=0.2, nx=6, ny=2, tip_load=-500, element="Q4")
    assert res["max_vm"] > 0
    assert res["elements"].shape[1] == 4

def test_q4_stiffness_symmetric():
    coords = np.array([[0.,0.],[1.,0.],[1.,1.],[0.,1.]])
    mat = Material()
    ke = element_stiffness_q4(coords, mat)
    assert ke.shape == (8,8)
    assert np.allclose(ke, ke.T)

def test_plate_with_hole():
    res = plate_with_hole_tension(L=1.0, R=0.15, n_r=5, n_th=12, sigma_far=100e6)
    assert res["max_vm"] > res["sigma_far"]   # concentration
    Kt = res["max_vm"] / res["sigma_far"]
    assert 1.5 < Kt < 6.0   # rough bounds for coarse mesh

def test_von_mises():
    assert abs(von_mises(100., 0., 0.) - 100) < 1e-9
    assert von_mises(0., 0., 50.) > 0

def test_mesh_size():
    nodes, elems = generate_rect_mesh(1.0, 0.5, 4, 2, element="CST")
    assert len(nodes) == 5*3 and len(elems) == 16
    nodes, elems = generate_rect_mesh(1.0, 0.5, 4, 2, element="Q4")
    assert len(elems) == 8 and elems.shape[1] == 4

if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
