import math
import json
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass, asdict
from itertools import combinations


# ============================================================
# 1. CONFIG
# ============================================================

@dataclass
class CriticalityConfigFSS:
    # lattice sizes for finite-size scaling
    sizes: tuple = (32, 48, 64, 96)

    dx: float = 1.0

    # TDGL dynamics
    gamma: float = 0.08
    beta: float = 0.08
    a0: float = 1.0
    lambda_c: float = 1.0
    T0: float = 0.03
    zeta: float = 1.0

    # simulation
    burn_in: int = 1800
    sample_steps: int = 2600
    sample_every: int = 10
    dtau: float = 1.0

    # scan
    lambda_min: float = 0.88
    lambda_max: float = 1.12
    lambda_points: int = 17

    # reproducibility
    seed: int = 42


# ============================================================
# 2. TDGL 2D EXPERIMENT
# ============================================================

class CoherenceFieldExperiment2DFSS:
    def __init__(self, cfg: CriticalityConfigFSS):
        self.cfg = cfg
        self.base_seed = cfg.seed

    def alpha_of_lambda(self, lam: float) -> float:
        return self.cfg.a0 * (self.cfg.lambda_c - lam)

    def initialize_field(self, L: int, rng: np.random.Generator) -> np.ndarray:
        return rng.normal(0.0, 1e-3, size=(L, L))

    def laplacian_periodic_2d(self, psi: np.ndarray) -> np.ndarray:
        lap_x = np.roll(psi, -1, axis=0) + np.roll(psi, 1, axis=0) - 2.0 * psi
        lap_y = np.roll(psi, -1, axis=1) + np.roll(psi, 1, axis=1) - 2.0 * psi
        return (lap_x + lap_y) / (self.cfg.dx ** 2)

    def step(self, psi: np.ndarray, alpha: float, rng: np.random.Generator) -> np.ndarray:
        lap = self.laplacian_periodic_2d(psi)
        force = lap - 2.0 * alpha * psi - 4.0 * self.cfg.beta * (psi ** 3)

        noise = rng.normal(
            loc=0.0,
            scale=math.sqrt(2.0 * self.cfg.gamma * self.cfg.T0),
            size=psi.shape
        )

        psi = psi + self.cfg.dtau * self.cfg.gamma * force + noise
        return psi

    def binder_cumulant(self, m_samples: np.ndarray) -> float:
        m2 = np.mean(m_samples ** 2)
        m4 = np.mean(m_samples ** 4)
        if m2 <= 1e-14:
            return 0.0
        return 1.0 - m4 / (3.0 * (m2 ** 2))

    def k_eff(self, alpha: float) -> float:
        if alpha >= 0:
            return 0.0
        return (self.cfg.zeta / (4.0 * math.pi)) * (abs(alpha) / (2.0 * self.cfg.beta))

    def connected_correlation_axis_average(self, psi: np.ndarray) -> np.ndarray:
        L = psi.shape[0]
        mean = psi.mean()
        corr_x = np.zeros(L)
        corr_y = np.zeros(L)

        for r in range(L):
            shifted_x = np.roll(psi, -r, axis=0)
            shifted_y = np.roll(psi, -r, axis=1)
            corr_x[r] = np.mean(psi * shifted_x) - mean**2
            corr_y[r] = np.mean(psi * shifted_y) - mean**2

        return 0.5 * (corr_x + corr_y)

    def estimate_correlation_length(self, corr: np.ndarray) -> float:
        c0 = corr[0]
        if c0 <= 0:
            return 0.0

        norm = corr / c0
        target = math.exp(-1.0)

        for r in range(1, len(norm) // 2):
            if norm[r] < target:
                return float(r * self.cfg.dx)

        return float((len(norm) // 2) * self.cfg.dx)

    def energy_density(self, psi: np.ndarray, alpha: float) -> float:
        dx_field = (np.roll(psi, -1, axis=0) - psi) / self.cfg.dx
        dy_field = (np.roll(psi, -1, axis=1) - psi) / self.cfg.dx
        grad2 = np.mean(dx_field**2 + dy_field**2)
        return float(0.5 * grad2 + alpha * np.mean(psi**2) + self.cfg.beta * np.mean(psi**4))

    def run_at_lambda_and_size(self, lam: float, L: int, seed_offset: int = 0) -> dict:
        alpha = self.alpha_of_lambda(lam)
        rng = np.random.default_rng(self.base_seed + 1000 * L + seed_offset)
        psi = self.initialize_field(L, rng)

        for _ in range(self.cfg.burn_in):
            psi = self.step(psi, alpha, rng)

        magnetizations = []
        abs_magnetizations = []
        energies = []
        corr_accum = np.zeros(L)
        corr_count = 0

        for t in range(self.cfg.sample_steps):
            psi = self.step(psi, alpha, rng)

            if t % self.cfg.sample_every == 0:
                m = float(np.mean(psi))
                magnetizations.append(m)
                abs_magnetizations.append(abs(m))
                energies.append(self.energy_density(psi, alpha))

                corr_accum += self.connected_correlation_axis_average(psi)
                corr_count += 1

        magnetizations = np.array(magnetizations)
        abs_magnetizations = np.array(abs_magnetizations)
        energies = np.array(energies)
        corr_mean = corr_accum / max(corr_count, 1)

        susceptibility = float((L ** 2) * np.var(magnetizations) / max(self.cfg.T0, 1e-12))

        return {
            "L": int(L),
            "lambda": float(lam),
            "alpha": float(alpha),
            "order_parameter": float(np.mean(abs_magnetizations)),
            "susceptibility": susceptibility,
            "binder": float(self.binder_cumulant(magnetizations)),
            "correlation_length": float(self.estimate_correlation_length(corr_mean)),
            "energy_density": float(np.mean(energies)),
            "k_eff": float(self.k_eff(alpha)),
        }

    def scan_all_sizes(self):
        lambdas = np.linspace(self.cfg.lambda_min, self.cfg.lambda_max, self.cfg.lambda_points)
        results = {int(L): [] for L in self.cfg.sizes}

        for L in self.cfg.sizes:
            print(f"\n=== Scanning L={L} ===")
            for lam in lambdas:
                res = self.run_at_lambda_and_size(float(lam), int(L))
                results[int(L)].append(res)
                print(
                    f"L={L:3d} | λ={res['lambda']:.4f} | α={res['alpha']:.4f} | "
                    f"m={res['order_parameter']:.5f} | χ={res['susceptibility']:.5f} | "
                    f"U4={res['binder']:.5f} | ξ={res['correlation_length']:.3f} | "
                    f"k={res['k_eff']:.5f}"
                )
        return results


# ============================================================
# 3. ANALYSIS
# ============================================================

def get_series(results_by_L, observable):
    series = {}
    for L, items in results_by_L.items():
        lambdas = np.array([r["lambda"] for r in items], dtype=float)
        values = np.array([r[observable] for r in items], dtype=float)
        series[int(L)] = (lambdas, values)
    return series

def estimate_lambda_c_from_binder_crossings(results_by_L):
    """
    Approximate λ_c from pairwise sign changes of Binder differences.
    Linear interpolation between nearest scan points.
    """
    binder_series = get_series(results_by_L, "binder")
    crossings = []

    sizes = sorted(binder_series.keys())
    for L1, L2 in combinations(sizes, 2):
        x1, u1 = binder_series[L1]
        x2, u2 = binder_series[L2]

        if len(x1) != len(x2) or not np.allclose(x1, x2):
            continue

        diff = u1 - u2

        for i in range(len(diff) - 1):
            if diff[i] == 0:
                crossings.append(float(x1[i]))
            elif diff[i] * diff[i + 1] < 0:
                # linear interpolation
                x_left, x_right = x1[i], x1[i + 1]
                y_left, y_right = diff[i], diff[i + 1]
                xc = x_left - y_left * (x_right - x_left) / (y_right - y_left)
                crossings.append(float(xc))

    if len(crossings) == 0:
        return None, []

    return float(np.mean(crossings)), crossings

def estimate_lambda_c_from_chi_peaks(results_by_L):
    peaks = []
    for L, items in results_by_L.items():
        chis = np.array([r["susceptibility"] for r in items], dtype=float)
        lambdas = np.array([r["lambda"] for r in items], dtype=float)
        idx = int(np.argmax(chis))
        peaks.append((int(L), float(lambdas[idx]), float(chis[idx])))
    if not peaks:
        return None, []
    lambda_est = float(np.mean([p[1] for p in peaks]))
    return lambda_est, peaks

def estimate_beta_exponent_fss(results_by_L, lambda_c_est):
    """
    Crude estimate using largest lattice only:
    log(m) ~ beta_exp * log(λ - λc)
    """
    largest_L = max(results_by_L.keys())
    items = results_by_L[largest_L]

    pts = []
    for r in items:
        dl = r["lambda"] - lambda_c_est
        m = r["order_parameter"]
        if dl > 0 and m > 1e-10:
            pts.append((dl, m))

    if len(pts) < 3:
        return None

    x = np.log(np.array([p[0] for p in pts[:5]]))
    y = np.log(np.array([p[1] for p in pts[:5]]))
    slope, intercept = np.polyfit(x, y, 1)
    return float(slope), float(intercept), int(largest_L)

def estimate_nu_exponent_fss(results_by_L, lambda_c_est):
    """
    Crude estimate using largest lattice only:
    log(ξ) ~ -nu * log|λ - λc|
    """
    largest_L = max(results_by_L.keys())
    items = results_by_L[largest_L]

    pts = []
    for r in items:
        dl = abs(r["lambda"] - lambda_c_est)
        xi = r["correlation_length"]
        if dl > 1e-10 and xi > 1e-10:
            pts.append((dl, xi))

    if len(pts) < 4:
        return None

    pts = sorted(pts, key=lambda z: z[0])[:6]
    x = np.log(np.array([p[0] for p in pts]))
    y = np.log(np.array([p[1] for p in pts]))
    slope, intercept = np.polyfit(x, y, 1)
    return float(-slope), float(intercept), int(largest_L)


# ============================================================
# 4. PLOTS
# ============================================================

def plot_binder_crossing(results_by_L, output_prefix="criticality_fss_2d"):
    binder_series = get_series(results_by_L, "binder")

    plt.figure(figsize=(8, 6))
    for L in sorted(binder_series.keys()):
        lambdas, binder = binder_series[L]
        plt.plot(lambdas, binder, marker="o", label=f"L={L}")

    plt.title("Binder Cumulant Crossing")
    plt.xlabel("λ")
    plt.ylabel("U4")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{output_prefix}_binder_crossing.png", dpi=220)
    plt.close()

def plot_susceptibility_by_size(results_by_L, output_prefix="criticality_fss_2d"):
    chi_series = get_series(results_by_L, "susceptibility")

    plt.figure(figsize=(8, 6))
    for L in sorted(chi_series.keys()):
        lambdas, chi = chi_series[L]
        plt.plot(lambdas, chi, marker="o", label=f"L={L}")

    plt.title("Susceptibility by Lattice Size")
    plt.xlabel("λ")
    plt.ylabel("χ")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{output_prefix}_susceptibility.png", dpi=220)
    plt.close()

def plot_order_parameter_by_size(results_by_L, output_prefix="criticality_fss_2d"):
    order_series = get_series(results_by_L, "order_parameter")

    plt.figure(figsize=(8, 6))
    for L in sorted(order_series.keys()):
        lambdas, order = order_series[L]
        plt.plot(lambdas, order, marker="o", label=f"L={L}")

    plt.title("Order Parameter by Lattice Size")
    plt.xlabel("λ")
    plt.ylabel("<|Ψ|>")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{output_prefix}_order_parameter.png", dpi=220)
    plt.close()

def plot_correlation_length_by_size(results_by_L, output_prefix="criticality_fss_2d"):
    xi_series = get_series(results_by_L, "correlation_length")

    plt.figure(figsize=(8, 6))
    for L in sorted(xi_series.keys()):
        lambdas, xi = xi_series[L]
        plt.plot(lambdas, xi, marker="o", label=f"L={L}")

    plt.title("Correlation Length Proxy by Lattice Size")
    plt.xlabel("λ")
    plt.ylabel("ξ")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{output_prefix}_correlation_length.png", dpi=220)
    plt.close()


# ============================================================
# 5. MAIN
# ============================================================

def main():
    cfg = CriticalityConfigFSS()
    exp = CoherenceFieldExperiment2DFSS(cfg)

    results_by_L = exp.scan_all_sizes()

    lambda_c_binder, binder_crossings = estimate_lambda_c_from_binder_crossings(results_by_L)
    lambda_c_chi, chi_peaks = estimate_lambda_c_from_chi_peaks(results_by_L)

    # prefer Binder estimate if available
    lambda_c_est = lambda_c_binder if lambda_c_binder is not None else lambda_c_chi

    beta_fit = estimate_beta_exponent_fss(results_by_L, lambda_c_est) if lambda_c_est is not None else None
    nu_fit = estimate_nu_exponent_fss(results_by_L, lambda_c_est) if lambda_c_est is not None else None

    summary = {
        "config": asdict(cfg),
        "lambda_c_input": cfg.lambda_c,
        "lambda_c_est_from_binder": lambda_c_binder,
        "binder_crossings": binder_crossings,
        "lambda_c_est_from_chi_peaks": lambda_c_chi,
        "chi_peaks": chi_peaks,
        "lambda_c_est_final": lambda_c_est,
        "beta_exp_fit": {
            "value": beta_fit[0],
            "intercept": beta_fit[1],
            "L_used": beta_fit[2],
        } if beta_fit is not None else None,
        "nu_exp_fit": {
            "value": nu_fit[0],
            "intercept": nu_fit[1],
            "L_used": nu_fit[2],
        } if nu_fit is not None else None,
        "results_by_L": results_by_L,
    }

    with open("criticality_fss_2d_results.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    plot_binder_crossing(results_by_L)
    plot_susceptibility_by_size(results_by_L)
    plot_order_parameter_by_size(results_by_L)
    plot_correlation_length_by_size(results_by_L)

    print("\n=== 2D FINITE-SIZE SCALING SUMMARY ===")
    print(f"Input λc: {cfg.lambda_c:.6f}")

    if lambda_c_binder is not None:
        print(f"Estimated λc from Binder crossings: {lambda_c_binder:.6f}")
    else:
        print("Estimated λc from Binder crossings: not found")

    if lambda_c_chi is not None:
        print(f"Estimated λc from susceptibility peaks: {lambda_c_chi:.6f}")
    else:
        print("Estimated λc from susceptibility peaks: not found")

    if lambda_c_est is not None:
        print(f"Final λc estimate: {lambda_c_est:.6f}")

    if beta_fit is not None:
        print(f"Estimated beta exponent (largest L): {beta_fit[0]:.4f}")
    else:
        print("Estimated beta exponent: insufficient data")

    if nu_fit is not None:
        print(f"Estimated nu exponent (largest L): {nu_fit[0]:.4f}")
    else:
        print("Estimated nu exponent: insufficient data")

    print("\nSaved files:")
    print(" - criticality_fss_2d_results.json")
    print(" - criticality_fss_2d_binder_crossing.png")
    print(" - criticality_fss_2d_susceptibility.png")
    print(" - criticality_fss_2d_order_parameter.png")
    print(" - criticality_fss_2d_correlation_length.png")


if __name__ == "__main__":
    main()