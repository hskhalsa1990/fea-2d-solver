# Mesh Refinement Strategies — Investigation Notes

Numerical experiments on the educational 2D FEA solver (CST / Q4, cantilever plate, Kirsch hole).

## 1. Uniform h-refinement (cantilever, tip load −1000 N)

| Element | nx | ny | DOF | max σ_vm (MPa) | max \|u\| (mm) |
|---------|----|----|-----|----------------|----------------|
| CST | 4 | 2 | 30 | 5.54 | 0.076 |
| CST | 8 | 2 | 54 | 7.64 | 0.124 |
| CST | 16 | 3 | 136 | 10.62 | 0.181 |
| CST | 32 | 6 | 462 | 13.79 | 0.226 |
| Q4 | 4 | 2 | 30 | 3.98 | 0.151 |
| Q4 | 8 | 2 | 54 | 5.78 | 0.209 |
| Q4 | 16 | 3 | 136 | 8.75 | 0.235 |
| Q4 | 32 | 6 | 462 | 11.74 | 0.244 |

- Displacement converges faster than peak stress (expected for CST/Q4).
- Relative change in max σ_vm between successive meshes drops toward a few percent at higher DOF.
- See: `refine_uniform_cst_q4.png`, `refine_relative_change.png`.

## 2. Aspect ratio (fixed L=1, H=0.2, CST)

| nx | ny | AR = Δx/Δy | max σ_vm | max \|u\| |
|----|----|------------|----------|-----------|
| 8 | 8 | 5.0 | 10.35 | 0.166 |
| 12 | 6 | 2.5 | 11.73 | 0.195 |
| 16 | 4 | 1.25 | 11.61 | 0.197 |
| 24 | 3 | 0.62 | 11.08 | 0.190 |
| 32 | 2 | 0.31 | 8.96 | 0.156 |

- Best stress/displacement behaviour near AR ≈ 1–2.5.
- Very flat elements (AR ≪ 1) under-predict peak stress on this mesh family.
- See: `refine_aspect_ratio.png`.

## 3. Kirsch hole — radial vs angular

| n_r | n_θ | Kt (num) |
|-----|-----|----------|
| 4 | 12–24 | ≈ 1.96 |
| 8 | 12–24 | ≈ 2.43 |
| 12 | 12–24 | ≈ 2.70 |

- Theory for infinite plate: **Kt = 3**.
- **Radial** refinement dominates improvement of Kt.
- Angular refinement beyond ~16 has small effect on this mesh.
- See: `refine_kirsch_radial_angular.png`.

## 4. DOF efficiency vs fine reference (CST nx=40, ny=8)

Error in max σ_vm falls with DOF for both CST and Q4; CST reaches lower error at the highest DOF tested on this structured family (problem- and mesh-dependent).

See: `refine_dof_efficiency.png`.

## Practical recommendations

1. **Start** with nx=12–16, ny=4 (plate) or n_r=8–12, n_θ=16–24 (hole).
2. **Refine uniformly** until relative change in max von Mises is below ~2–3%.
3. Keep **element aspect ratio** roughly in [0.5, 3].
4. For **stress concentrations**, put more divisions near the hole / re-entrant corners first.
5. Use **nodal stress averaging** for smoother contour plots; still refine the mesh for quantitative peaks.
6. CST is fine for coarse studies; Q4 often gives smoother fields for similar nx.

Diagrams: `refine_strategy_guide.png` and the other `refine_*.png` files under ``.
