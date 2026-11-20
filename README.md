# 2D Finite Element Analysis Solver — v2

Linear-elastic 2D FEA with CST & Q4 elements, beam, truss, plate-with-hole (Kirsch), nodal stress averaging, Streamlit dashboard and ML surrogate.

---

## Quick Start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
streamlit run app_streamlit.py
pytest test_fea.py -v
```

---

## Architecture

![Architecture](architecture.png)

---

## Diagram Gallery

### Stress Components (σx, σy, τxy, von Mises)
![Stress panel](stress_components_panel.png)

### Displacement Magnitude Field
![Displacement](displacement_field.png)

### CST vs Q4 Side-by-Side
![CST vs Q4](cst_vs_q4_sidebyside.png)

### Cantilever — CST / Q4 Contours
![CST](plate_von_mises_cst.png)
![Q4](plate_von_mises_q4.png)

### Cantilever — Von Mises Contour
![Contour](plate_von_mises_contour.png)

### Deformed Mesh
![Deformed](plate_deformed_mesh.png)

### Plate with Circular Hole (Kirsch)
![Hole](plate_with_hole.png)

### Kirsch Kt Convergence
![Kt](kirsch_kt_convergence.png)

### Mesh Convergence (CST vs Q4)
![Convergence](mesh_convergence.png)

### Convergence Rate
![Rate](convergence_rate.png)

### Truss Axial Forces + Deformed Shape
![Truss](truss_forces_deformed.png)

### Beam Deflection Family
![Beam family](beam_deflection_family.png)

### Beam Validation
![Beam](beam_validation.png)

### BC & Load Schematic
![Schematic](bc_load_schematic.png)

### Feature Coverage
![Features](feature_coverage.png)

---

## Mesh Refinement Investigation

See **[MESH_REFINEMENT.md](MESH_REFINEMENT.md)** for full tables and conclusions.

![Strategy guide](refine_strategy_guide.png)

![Uniform CST vs Q4](refine_uniform_cst_q4.png)

![Relative change](refine_relative_change.png)

![Aspect ratio](refine_aspect_ratio.png)

![Kirsch radial/angular](refine_kirsch_radial_angular.png)

![DOF efficiency](refine_dof_efficiency.png)

## Validation

| Solver | Benchmark | Result |
|--------|-----------|--------|
| Truss | Single bar axial | Machine precision |
| Beam | Cantilever tip load | Exact with 1 element |
| Plate CST/Q4 | Mesh convergence | Monotonic stabilisation |
| Hole | Kirsch Kt ≈ 3 | Recovered with refinement |

\[
\delta_{\text{beam}} = \frac{P L^3}{3 E I}
\]

---

## Modules

| File | Role |
|------|------|
| `fea_truss.py` | 2D truss |
| `fea_beam.py` | Euler-Bernoulli beam |
| `fea_plate.py` | CST + Q4, hole mesh, nodal averaging |
| `mesh_utils.py` | Simple rectangular mesh helpers |
| `ml_surrogate.py` | Max-stress predictor |
| `app_streamlit.py` | 5-tab interactive dashboard |
| `test_fea.py` | 8 unit tests |

---

## Layout (completely flat — no sub-folders)

```
.
├── README.md  MESH_REFINEMENT.md  LICENSE  requirements.txt
├── fea_truss.py  fea_beam.py  fea_plate.py  mesh_utils.py
├── ml_surrogate.py  app_streamlit.py  test_fea.py
├── FEA_2D_Theory_Validation.ipynb
└── 23 PNG diagrams (architecture.png, plate_with_hole.png, ...)
```

**Tags:** `fea` `finite-element` `plane-stress` `cst` `q4` `kirsch` `truss` `beam` `mechanical-engineering` `python` `streamlit`


## License

MIT — see [LICENSE](LICENSE).
