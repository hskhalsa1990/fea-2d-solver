"""scikit-learn surrogate model: geometry/load -> max von Mises stress."""
from __future__ import annotations
import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_percentage_error
from fea_plate import Material, generate_rect_mesh, solve_plate

def generate_training_data(n_samples: int = 120, seed: int = 42) -> tuple:
    """Vary L, H, nx, tip load, E and collect max_vm."""
    rng = np.random.default_rng(seed)
    X, y = [], []
    for _ in range(n_samples):
        L = rng.uniform(0.5, 2.0)
        H = rng.uniform(0.1, 0.4)
        nx = int(rng.integers(8, 24))
        ny = max(2, nx // 5)
        E = rng.uniform(70e9, 210e9)   # Al to steel
        load = rng.uniform(-5000, -200)
        mat = Material(E=E, nu=0.3, thickness=0.01)
        nodes, elements = generate_rect_mesh(L, H, nx, ny, element="CST")
        fixed = [2*i for i, (x,_) in enumerate(nodes) if abs(x)<1e-9] + \
                [2*i+1 for i, (x,_) in enumerate(nodes) if abs(x)<1e-9]
        right = [i for i,(x,_) in enumerate(nodes) if abs(x-L)<1e-9]
        loads = {i: (0.0, load/len(right)) for i in right}
        try:
            res = solve_plate(nodes, elements, mat, loads, fixed)
            X.append([L, H, nx, abs(load), E/1e9])
            y.append(res["max_vm"] / 1e6)  # MPa
        except Exception:
            continue
    return np.array(X), np.array(y)

def train_surrogate(n_samples: int = 120, seed: int = 42):
    X, y = generate_training_data(n_samples, seed)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=seed)
    model = GradientBoostingRegressor(n_estimators=80, max_depth=4, random_state=seed)
    model.fit(Xtr, ytr)
    pred = model.predict(Xte)
    metrics = {
        "r2": float(r2_score(yte, pred)),
        "mape": float(mean_absolute_percentage_error(yte, pred)),
        "n_train": len(Xtr),
        "n_test": len(Xte),
    }
    return model, metrics, (Xte, yte, pred)

def predict_max_stress(model, L: float, H: float, nx: int, load: float, E_GPa: float) -> float:
    """Return predicted max von Mises in MPa."""
    return float(model.predict([[L, H, nx, abs(load), E_GPa]])[0])
