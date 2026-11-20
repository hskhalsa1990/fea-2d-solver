"""Streamlit dashboard — 2D FEA Solver (upgraded)."""
from __future__ import annotations
import numpy as np
import streamlit as st
import plotly.graph_objects as go
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))

from fea_plate import (Material, generate_rect_mesh, solve_plate, cantilever_plate_mesh,
                       plate_with_hole_tension, von_mises)
from fea_beam import BeamElement, solve_beam, cantilever_analytical
from fea_truss import TrussElement, solve_truss, assemble_truss, apply_bc_and_solve, element_forces_truss
from ml_surrogate import train_surrogate, predict_max_stress

st.set_page_config(page_title="2D FEA Solver v2", layout="wide", page_icon="🔧")
st.title("2D Finite Element Analysis Solver — v2")
st.caption("CST + Q4 · Beam · Truss · Plate-with-hole · Nodal stress averaging · ML Surrogate")

tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ["Plate (CST/Q4)", "Plate with Hole", "Beam", "Truss", "ML Surrogate"]
)

def plot_mesh_stress(nodes, elems, stresses, title, scale_disp=0.0, disp=None):
    """Plotly mesh + element-center stress colour."""
    fig = go.Figure()
    centers, vm = [], []
    for i, el in enumerate(elems):
        pts = nodes[el]
        if disp is not None and scale_disp > 0:
            pts = pts + scale_disp * disp[el]
        xs = list(pts[:,0]) + [pts[0,0]]
        ys = list(pts[:,1]) + [pts[0,1]]
        fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines",
            line=dict(color="#444", width=0.7), showlegend=False, hoverinfo="skip"))
        centers.append(pts.mean(0))
        vm.append(stresses[i, 3] / 1e6)
    centers = np.array(centers)
    vm = np.array(vm)
    fig.add_trace(go.Scatter(
        x=centers[:,0], y=centers[:,1], mode="markers",
        marker=dict(size=7, color=vm, colorscale="Jet",
                    colorbar=dict(title="σ_vm (MPa)"), cmin=0, cmax=max(vm.max(), 1e-9)),
        text=[f"{v:.1f} MPa" for v in vm], hoverinfo="text", name="von Mises"))
    fig.update_layout(title=title, xaxis_title="x (m)", yaxis_title="y (m)",
                      yaxis_scaleanchor="x", height=480, margin=dict(l=20,r=20,t=40,b=20),
                      template="plotly_white")
    return fig

# -------------------- Plate CST / Q4 --------------------
with tab1:
    st.header("Cantilever Plate — CST or Q4")
    c1, c2, c3 = st.columns(3)
    with c1:
        L = st.number_input("Length L (m)", 0.2, 5.0, 1.0, 0.1, key="pL")
        H = st.number_input("Height H (m)", 0.05, 1.0, 0.25, 0.05, key="pH")
        t = st.number_input("Thickness (m)", 0.001, 0.1, 0.01, 0.001, key="pt")
    with c2:
        nx = st.slider("Divisions x", 4, 40, 16, key="pnx")
        ny = st.slider("Divisions y", 2, 12, 4, key="pny")
        elem_type = st.selectbox("Element", ["CST", "Q4"], key="pelem")
    with c3:
        E = st.number_input("E (GPa)", 10.0, 400.0, 210.0, 10.0, key="pE") * 1e9
        nu = st.number_input("Poisson ν", 0.0, 0.49, 0.3, 0.01, key="pnu")
        tip_load = st.number_input("Tip load (N)", -20000.0, 0.0, -2000.0, 100.0, key="pload")
    if st.button("Solve Plate", type="primary", key="pbtn"):
        with st.spinner("Solving..."):
            res = cantilever_plate_mesh(L=L, H=H, nx=nx, ny=ny, E=E, nu=nu, t=t,
                                        tip_load=tip_load, element=elem_type)
            st.session_state["plate"] = res
    if "plate" in st.session_state:
        res = st.session_state["plate"]
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Max von Mises", f"{res['max_vm']/1e6:.2f} MPa")
        m2.metric("Max |u|", f"{res['max_disp']*1e3:.3f} mm")
        m3.metric("Elements", len(res["elements"]))
        m4.metric("Nodes", len(res["nodes"]))
        scale = st.slider("Deformation scale", 0.0, 300.0, 50.0, key="pscale")
        st.plotly_chart(plot_mesh_stress(
            res["nodes"], res["elements"], res["stresses"],
            f"Cantilever ({elem_type}) — von Mises", scale, res["displacements"]),
            use_container_width=True)

# -------------------- Plate with hole --------------------
with tab2:
    st.header("Quarter-Plate with Circular Hole (Kirsch)")
    st.write("Remote tension · symmetry BCs · theoretical stress concentration Kt ≈ 3")
    c1, c2, c3 = st.columns(3)
    with c1:
        Lh = st.number_input("Outer size L (m)", 0.5, 3.0, 1.0, key="hL")
        Rh = st.number_input("Hole radius R (m)", 0.05, 0.4, 0.15, key="hR")
    with c2:
        nr = st.slider("Radial divisions", 4, 16, 8, key="hnr")
        nth = st.slider("Angular divisions", 8, 32, 20, key="hnth")
    with c3:
        sig = st.number_input("σ_far (MPa)", 10.0, 500.0, 100.0, key="hsig") * 1e6
        Eh = st.number_input("E (GPa)", 10.0, 400.0, 210.0, key="hE") * 1e9
    if st.button("Solve Hole Problem", type="primary"):
        with st.spinner("Meshing & solving..."):
            res = plate_with_hole_tension(L=Lh, R=Rh, n_r=nr, n_th=nth,
                                          sigma_far=sig, E=Eh)
            st.session_state["hole"] = res
    if "hole" in st.session_state:
        res = st.session_state["hole"]
        Kt_num = res["max_vm"] / res["sigma_far"]
        m1, m2, m3 = st.columns(3)
        m1.metric("Max von Mises", f"{res['max_vm']/1e6:.1f} MPa")
        m2.metric("Kt (numerical)", f"{Kt_num:.2f}")
        m3.metric("Kt (theory ∞ plate)", "3.00")
        st.plotly_chart(plot_mesh_stress(
            res["nodes"], res["elements"], res["stresses"],
            "Plate with Hole — von Mises (quarter model)"), use_container_width=True)

# -------------------- Beam --------------------
with tab3:
    st.header("Euler-Bernoulli Beam — Cantilever")
    c1, c2 = st.columns(2)
    with c1:
        Lb = st.number_input("Length (m)", 0.2, 5.0, 1.0, key="bL")
        Eb = st.number_input("E (GPa)", 10.0, 400.0, 210.0, key="bE") * 1e9
        Ib = st.number_input("I (m⁴)", 1e-8, 1e-4, 1e-6, format="%.2e", key="bI")
    with c2:
        Pb = st.number_input("Tip load P (N)", -10000.0, 0.0, -1000.0, key="bP")
        n_el = st.slider("Elements", 1, 20, 4, key="bne")
    if st.button("Solve Beam"):
        xs = np.linspace(0, Lb, n_el+1)
        nodes = np.column_stack([xs, np.zeros_like(xs)])
        elems = [BeamElement(i, i+1, E=Eb, I=Ib) for i in range(n_el)]
        res = solve_beam(nodes, elems, loads={n_el: (0., Pb, 0.)}, fixed_dofs=[0,1,2])
        u_exact = cantilever_analytical(Lb, abs(Pb), Eb, Ib) * np.sign(Pb)
        tip = res["displacements"][-1, 1]
        st.metric("Tip deflection FEA", f"{tip*1e3:.4f} mm")
        st.metric("Analytical", f"{u_exact*1e3:.4f} mm")
        st.metric("Relative error", f"{abs(tip-u_exact)/abs(u_exact)*100:.3f} %")
        xa = np.linspace(0, Lb, 80)
        ya = (Pb/(6*Eb*Ib)) * (3*Lb*xa**2 - xa**3) * 1e3
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=nodes[:,0], y=res["displacements"][:,1]*1e3,
                                 mode="lines+markers", name="FEA"))
        fig.add_trace(go.Scatter(x=xa, y=ya, mode="lines", name="Analytical", line=dict(dash="dash")))
        fig.update_layout(title="Beam Deflection", xaxis_title="x (m)", yaxis_title="uy (mm)",
                          height=400, template="plotly_white")
        st.plotly_chart(fig, use_container_width=True)

# -------------------- Truss --------------------
with tab4:
    st.header("2D Truss — Simple Bridge")
    if st.button("Solve 3-bar Truss"):
        nodes = np.array([[0.,0.], [1.,0.], [0.5, 0.866]])
        elems = [
            TrussElement(0, 2, E=210e9, A=0.001),
            TrussElement(1, 2, E=210e9, A=0.001),
            TrussElement(0, 1, E=210e9, A=0.002),
        ]
        res = solve_truss(nodes, elems, loads={2: (0., -5000.)}, fixed_nodes=[0,1])
        st.write("Displacements (m):"); st.dataframe(res["displacements"])
        st.write("Axial forces (N):", np.round(res["axial_forces"], 1))
        scale = 150
        def_n = nodes + scale * res["displacements"]
        fig = go.Figure()
        for e in elems:
            fig.add_trace(go.Scatter(x=[nodes[e.n1,0], nodes[e.n2,0]],
                                     y=[nodes[e.n1,1], nodes[e.n2,1]],
                                     mode="lines", line=dict(color="gray", width=2), showlegend=False))
            fig.add_trace(go.Scatter(x=[def_n[e.n1,0], def_n[e.n2,0]],
                                     y=[def_n[e.n1,1], def_n[e.n2,1]],
                                     mode="lines", line=dict(color="crimson", width=3), showlegend=False))
        fig.add_trace(go.Scatter(x=nodes[:,0], y=nodes[:,1], mode="markers+text",
                                 text=["N0","N1","N2"], marker=dict(size=14, color="steelblue")))
        fig.update_layout(title="Truss (gray=original, red=deformed)", yaxis_scaleanchor="x",
                          height=420, template="plotly_white")
        st.plotly_chart(fig, use_container_width=True)

# -------------------- ML --------------------
with tab5:
    st.header("ML Surrogate — Max Stress Predictor")
    st.write("Gradient Boosting trained on FEA samples → predicts max von Mises (MPa).")
    if st.button("Train Surrogate (~20–40 s)"):
        with st.spinner("Generating FEA samples & training..."):
            model, metrics, (Xte, yte, pred) = train_surrogate(n_samples=80, seed=42)
            st.session_state["ml"] = (model, metrics, Xte, yte, pred)
    if "ml" in st.session_state:
        model, metrics, Xte, yte, pred = st.session_state["ml"]
        st.success(f"R² = {metrics['r2']:.3f}  |  MAPE = {metrics['mape']*100:.1f}%  "
                   f"|  n_train/test = {metrics['n_train']}/{metrics['n_test']}")
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=yte, y=pred, mode="markers", name="Predictions"))
        lim = [min(yte.min(), pred.min()), max(yte.max(), pred.max())]
        fig.add_trace(go.Scatter(x=lim, y=lim, mode="lines", name="Ideal", line=dict(dash="dash")))
        fig.update_layout(title="Predicted vs FEA max σ_vm (MPa)", xaxis_title="FEA", yaxis_title="ML",
                          height=400, template="plotly_white")
        st.plotly_chart(fig, use_container_width=True)
        st.subheader("Quick prediction")
        cols = st.columns(5)
        Lp = cols[0].number_input("L", 0.5, 2.0, 1.0, key="mlL")
        Hp = cols[1].number_input("H", 0.1, 0.4, 0.2, key="mlH")
        nxp = cols[2].number_input("nx", 8, 30, 16, key="mlnx")
        loadp = cols[3].number_input("|Load|", 200, 5000, 1000, key="mlload")
        Ep = cols[4].number_input("E (GPa)", 70, 210, 210, key="mlE")
        st.metric("Predicted max σ_vm", f"{predict_max_stress(model, Lp, Hp, nxp, loadp, Ep):.1f} MPa")
