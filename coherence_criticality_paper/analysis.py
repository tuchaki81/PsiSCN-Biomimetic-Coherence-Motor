import numpy as np
from itertools import combinations

def estimate_lambda_c_from_binder(results_by_L):
    crossings = []
    sizes = sorted(results_by_L.keys())

    for L1, L2 in combinations(sizes, 2):
        items1 = results_by_L[L1]
        items2 = results_by_L[L2]
        x1 = np.array([r["lambda"] for r in items1])
        x2 = np.array([r["lambda"] for r in items2])
        y1 = np.array([r["binder_mean"] for r in items1])
        y2 = np.array([r["binder_mean"] for r in items2])

        if len(x1) != len(x2) or not np.allclose(x1, x2):
            continue

        diff = y1 - y2
        for i in range(len(diff) - 1):
            if diff[i] * diff[i + 1] < 0:
                xl, xr = x1[i], x1[i + 1]
                yl, yr = diff[i], diff[i + 1]
                xc = xl - yl * (xr - xl) / (yr - yl)
                crossings.append(float(xc))

    if not crossings:
        return None, []
    return float(np.mean(crossings)), crossings

def estimate_lambda_c_from_chi(results_by_L):
    peaks = []
    for L, items in results_by_L.items():
        x = np.array([r["lambda"] for r in items])
        y = np.array([r["susceptibility_mean"] for r in items])
        idx = int(np.argmax(y))
        peaks.append((int(L), float(x[idx]), float(y[idx])))

    if not peaks:
        return None, []
    return float(np.mean([p[1] for p in peaks])), peaks

def collapse_quality(x_scaled, y_scaled, bins=18):
    x_scaled = np.asarray(x_scaled)
    y_scaled = np.asarray(y_scaled)

    xmin, xmax = np.min(x_scaled), np.max(x_scaled)
    if xmax - xmin < 1e-12:
        return np.inf

    edges = np.linspace(xmin, xmax, bins + 1)
    total_var = 0.0
    used = 0

    for i in range(bins):
        if i < bins - 1:
            mask = (x_scaled >= edges[i]) & (x_scaled < edges[i + 1])
        else:
            mask = (x_scaled >= edges[i]) & (x_scaled <= edges[i + 1])

        if np.sum(mask) >= 2:
            total_var += np.var(y_scaled[mask])
            used += 1

    return np.inf if used == 0 else total_var / used

def prepare_collapse_order(results_by_L, lambda_c_est, beta_exp, nu):
    xs, ys, labels = [], [], []
    for L, items in results_by_L.items():
        Lf = float(L)
        for r in items:
            xs.append((r["lambda"] - lambda_c_est) * (Lf ** (1.0 / nu)))
            ys.append(r["order_parameter_mean"] * (Lf ** (beta_exp / nu)))
            labels.append(L)
    return np.array(xs), np.array(ys), np.array(labels)

def prepare_collapse_chi(results_by_L, lambda_c_est, gamma_exp, nu):
    xs, ys, labels = [], [], []
    for L, items in results_by_L.items():
        Lf = float(L)
        for r in items:
            xs.append((r["lambda"] - lambda_c_est) * (Lf ** (1.0 / nu)))
            ys.append(r["susceptibility_mean"] / (Lf ** (gamma_exp / nu)))
            labels.append(L)
    return np.array(xs), np.array(ys), np.array(labels)

def optimize_order_collapse(results_by_L, lambda_c_est, cfg):
    best = None
    nu_grid = np.linspace(cfg.nu_min, cfg.nu_max, cfg.nu_points)
    beta_grid = np.linspace(cfg.beta_exp_min, cfg.beta_exp_max, cfg.beta_exp_points)

    for nu in nu_grid:
        for beta_exp in beta_grid:
            x, y, _ = prepare_collapse_order(results_by_L, lambda_c_est, beta_exp, nu)
            q = collapse_quality(x, y)
            if best is None or q < best["quality"]:
                best = {"nu": float(nu), "beta_exp": float(beta_exp), "quality": float(q)}
    return best

def optimize_chi_collapse(results_by_L, lambda_c_est, cfg):
    best = None
    nu_grid = np.linspace(cfg.nu_min, cfg.nu_max, cfg.nu_points)
    gamma_grid = np.linspace(cfg.gamma_exp_min, cfg.gamma_exp_max, cfg.gamma_exp_points)

    for nu in nu_grid:
        for gamma_exp in gamma_grid:
            x, y, _ = prepare_collapse_chi(results_by_L, lambda_c_est, gamma_exp, nu)
            q = collapse_quality(x, y)
            if best is None or q < best["quality"]:
                best = {"nu": float(nu), "gamma_exp": float(gamma_exp), "quality": float(q)}
    return best