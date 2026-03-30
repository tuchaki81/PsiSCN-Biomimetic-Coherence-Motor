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
class CriticalityConfigAdvanced:
    sizes: tuple = (24, 32, 48, 64)

    dx: float = 1.0

    # TDGL dynamics
    gamma: float = 0.08
    beta: float = 0.08
    a0: float = 1.0
    lambda_c: float = 1.0
    T0: float = 0.03
    zeta: float = 1.0

    # simulation
    burn_in: int = 1600
    sample_steps: int = 2600
    sample_every: int = 10
    dtau: float = 1.0

    # scan
    lambda_min: float = 0.90
    lambda_max: float = 1.10
    lambda_points: int = 15

    # repeats / seeds
    repeats: int = 6
    base_seed: int = 42

    # bootstrap
    bootstrap_samples: int = 100

    # collapse search ranges
    nu_min: float = 0.4
    nu_max: float = 1.4
    nu_points: int = 31

    beta_exp_min: float = 0.05
    beta_exp_max: float = 0.30
    beta_exp_points: int = 31

    gamma_exp_min: float = 0.5
    gamma_exp_max: float = 2.0
    gamma_exp_points: int = 31

    # autocorrelation
    max_lag_fraction: float = 0.25   # maximum lag as fraction of sample length
    autocorr_cutoff: float = 0.0     # stop tau_int sum when rho(lag) <= cutoff


# ============================================================
# 2. TDGL 2D EXPERIMENT
# ============================================================

class CoherenceFieldExperiment2DAdvanced:
    def __init__(self, cfg: CriticalityConfigAdvanced):
        self.cfg = cfg

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

    def connected_correlation_radial(self, psi: np.ndarray) -> np.ndarray:
        """
        Radial isotropic connected correlation using FFT:
        C(r) = <psi(x) psi(x+r)> - <psi>^2
        Then binned by radial distance.
        """
        L = psi.shape[0]
        mean = np.mean(psi)
        phi = psi - mean

        # periodic autocorrelation via FFT
        fphi = np.fft.fftn(phi)
        corr2d = np.fft.ifftn(np.abs(fphi) ** 2).real / (L * L)

        # coordinate grid of wrapped distances
        coords = np.arange(L)
        dx = np.minimum(coords, L - coords)
        X, Y = np.meshgrid(dx, dx, indexing="ij")
        R = np.sqrt(X**2 + Y**2)

        r_max = L // 2
        radial = np.zeros(r_max + 1)
        counts = np.zeros(r_max + 1)

        r_int = np.rint(R).astype(int)
        for i in range(L):
            for j in range(L):
                rr = r_int[i, j]
                if rr <= r_max:
                    radial[rr] += corr2d[i, j]
                    counts[rr] += 1

        counts = np.where(counts == 0, 1, counts)
        radial /= counts
        return radial

    def estimate_correlation_length(self, corr: np.ndarray) -> float:
        """
        Exponential proxy from radial connected correlation.
        """
        c0 = corr[0]
        if c0 <= 0:
            return 0.0

        norm = corr / c0
        target = math.exp(-1.0)

        for r in range(1, len(norm)):
            if norm[r] < target:
                return float(r * self.cfg.dx)

        return float((len(norm) - 1) * self.cfg.dx)

    def energy_density(self, psi: np.ndarray, alpha: float) -> float:
        dx_field = (np.roll(psi, -1, axis=0) - psi) / self.cfg.dx
        dy_field = (np.roll(psi, -1, axis=1) - psi) / self.cfg.dx
        grad2 = np.mean(dx_field**2 + dy_field**2)
        return float(0.5 * grad2 + alpha * np.mean(psi**2) + self.cfg.beta * np.mean(psi**4))

    def k_eff(self, alpha: float) -> float:
        if alpha >= 0:
            return 0.0
        return (self.cfg.zeta / (4.0 * math.pi)) * (abs(alpha) / (2.0 * self.cfg.beta))

    def integrated_autocorrelation_time(self, series: np.ndarray) -> float:
        """
        Estimate integrated autocorrelation time:
        tau_int = 1/2 + sum_{lag>=1} rho(lag)
        truncated when rho(lag) <= cutoff or max lag reached.
        """
        x = np.asarray(series, dtype=float)
        n = len(x)
        if n < 4:
            return 0.5

        x = x - np.mean(x)
        var = np.var(x)
        if var <= 1e-16:
            return 0.5

        max_lag = max(1, int(self.cfg.max_lag_fraction * n))
        tau_int = 0.5

        for lag in range(1, max_lag):
            c = np.dot(x[:-lag], x[lag:]) / (n - lag)
            rho = c / var
            if rho <= self.cfg.autocorr_cutoff:
                break
            tau_int += rho

        return max(tau_int, 0.5)

    def effective_standard_error(self, series: np.ndarray) -> float:
        """
        Correct stderr using effective sample size:
        N_eff ~ N / (2 tau_int)
        """
        x = np.asarray(series, dtype=float)
        n = len(x)
        if n < 2:
            return 0.0

        tau_int = self.integrated_autocorrelation_time(x)
        n_eff = max(1.0, n / (2.0 * tau_int))
        return float(np.std(x, ddof=1) / math.sqrt(n_eff))

    def run_single(self, lam: float, L: int, seed: int) -> dict:
        alpha = self.alpha_of_lambda(lam)
        rng = np.random.default_rng(seed)
        psi = self.initialize_field(L, rng)

        for _ in range(self.cfg.burn_in):
            psi = self.step(psi, alpha, rng)

        magnetizations = []
        abs_magnetizations = []
        energies = []
        corr_accum = None
        corr_count = 0

        for t in range(self.cfg.sample_steps):
            psi = self.step(psi, alpha, rng)

            if t % self.cfg.sample_every == 0:
                m = float(np.mean(psi))
                magnetizations.append(m)
                abs_magnetizations.append(abs(m))
                energies.append(self.energy_density(psi, alpha))

                corr = self.connected_correlation_radial(psi)
                if corr_accum is None:
                    corr_accum = np.zeros_like(corr)
                corr_accum += corr
                corr_count += 1

        magnetizations = np.array(magnetizations)
        abs_magnetizations = np.array(abs_magnetizations)
        energies = np.array(energies)
        corr_mean = corr_accum / max(corr_count, 1)

        tau_m = self.integrated_autocorrelation_time(magnetizations)
        tau_absm = self.integrated_autocorrelation_time(abs_magnetizations)

        chi = float((L ** 2) * np.var(magnetizations) / max(self.cfg.T0, 1e-12))

        return {
            "order_parameter": float(np.mean(abs_magnetizations)),
            "order_parameter_se_timecorr": self.effective_standard_error(abs_magnetizations),
            "susceptibility": chi,
            "binder": float(self.binder_cumulant(magnetizations)),
            "correlation_length": float(self.estimate_correlation_length(corr_mean)),
            "energy_density": float(np.mean(energies)),
            "energy_density_se_timecorr": self.effective_standard_error(energies),
            "k_eff": float(self.k_eff(alpha)),
            "tau_int_m": float(tau_m),
            "tau_int_absm": float(tau_absm),
        }

    def run_ensemble(self, lam: float, L: int) -> dict:
        ensemble = []
        for rep in range(self.cfg.repeats):
            seed = self.cfg.base_seed + 10000 * L + 100 * int(round(lam * 1000)) + rep
            ensemble.append(self.run_single(lam, L, seed))

        def mean_std(key):
            vals = np.array([e[key] for e in ensemble], dtype=float)
            return float(np.mean(vals)), float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0

        order_m, order_s = mean_std("order_parameter")
        chi_m, chi_s = mean_std("susceptibility")
        binder_m, binder_s = mean_std("binder")
        xi_m, xi_s = mean_std("correlation_length")
        e_m, e_s = mean_std("energy_density")
        k_m, k_s = mean_std("k_eff")
        tau_m_m, tau_m_s = mean_std("tau_int_m")
        tau_absm_m, tau_absm_s = mean_std("tau_int_absm")

        return {
            "L": int(L),
            "lambda": float(lam),
            "alpha": float(self.alpha_of_lambda(lam)),
            "ensemble": ensemble,
            "order_parameter_mean": order_m,
            "order_parameter_std": order_s,
            "susceptibility_mean": chi_m,
            "susceptibility_std": chi_s,
            "binder_mean": binder_m,
            "binder_std": binder_s,
            "correlation_length_mean": xi_m,
            "correlation_length_std": xi_s,
            "energy_density_mean": e_m,
            "energy_density_std": e_s,
            "k_eff_mean": k_m,
            "k_eff_std": k_s,
            "tau_int_m_mean": tau_m_m,
            "tau_int_m_std": tau_m_s,
            "tau_int_absm_mean": tau_absm_m,
            "tau_int_absm_std": tau_absm_s,
        }

    def scan_all_sizes(self):
        lambdas = np.linspace(self.cfg.lambda_min, self.cfg.lambda_max, self.cfg.lambda_points)
        results = {int(L): [] for L in self.cfg.sizes}

        for L in self.cfg.sizes:
            print(f"\n=== Scanning L={L} ===")
            for lam in lambdas:
                res = self.run_ensemble(float(lam), int(L))
                results[int(L)].append(res)
                print(
                    f"L={L:3d} | λ={res['lambda']:.4f} | "
                    f"m={res['order_parameter_mean']:.5f}±{res['order_parameter_std']:.5f} | "
                    f"χ={res['susceptibility_mean']:.5f}±{res['susceptibility_std']:.5f} | "
                    f"U4={res['binder_mean']:.5f}±{res['binder_std']:.5f} | "
                    f"τm={res['tau_int_m_mean']:.3f}"
                )
        return results


# ============================================================
# 3. SERIES / HELPERS
# ============================================================

def mean_series(results_by_L, key):
    out = {}
    for L, items in results_by_L.items():
        x = np.array([r["lambda"] for r in items], dtype=float)
        y = np.array([r[key] for r in items], dtype=float)
        out[int(L)] = (x, y)
    return out

def bootstrap_resample_results(results_by_L, rng):
    boot = {}
    for L, items in results_by_L.items():
        boot[L] = []
        for item in items:
            ensemble = item["ensemble"]
            n = len(ensemble)
            idx = rng.integers(0, n, size=n)
            sampled = [ensemble[i] for i in idx]

            def mean_of(key):
                return float(np.mean([s[key] for s in sampled]))

            boot[L].append({
                "L": item["L"],
                "lambda": item["lambda"],
                "alpha": item["alpha"],
                "order_parameter_mean": mean_of("order_parameter"),
                "susceptibility_mean": mean_of("susceptibility"),
                "binder_mean": mean_of("binder"),
                "correlation_length_mean": mean_of("correlation_length"),
                "energy_density_mean": mean_of("energy_density"),
                "k_eff_mean": mean_of("k_eff"),
            })
    return boot


# ============================================================
# 4. λc ESTIMATION
# ============================================================

def estimate_lambda_c_from_binder(results_by_L):
    binders = mean_series(results_by_L, "binder_mean")
    crossings = []

    sizes = sorted(binders.keys())
    for L1, L2 in combinations(sizes, 2):
        x1, u1 = binders[L1]
        x2, u2 = binders[L2]
        if len(x1) != len(x2) or not np.allclose(x1, x2):
            continue

        diff = u1 - u2
        for i in range(len(diff) - 1):
            if diff[i] == 0:
                crossings.append(float(x1[i]))
            elif diff[i] * diff[i + 1] < 0:
                xl, xr = x1[i], x1[i + 1]
                yl, yr = diff[i], diff[i + 1]
                xc = xl - yl * (xr - xl) / (yr - yl)
                crossings.append(float(xc))

    if not crossings:
        return None, []
    return float(np.mean(crossings)), crossings

def estimate_lambda_c_from_chi(results_by_L):
    chis = mean_series(results_by_L, "susceptibility_mean")
    peaks = []
    for L, (x, y) in chis.items():
        idx = int(np.argmax(y))
        peaks.append((int(L), float(x[idx]), float(y[idx])))

    if not peaks:
        return None, []
    return float(np.mean([p[1] for p in peaks])), peaks


# ============================================================
# 5. COLLAPSE
# ============================================================

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
            lam = r["lambda"]
            m = r["order_parameter_mean"]
            xs.append((lam - lambda_c_est) * (Lf ** (1.0 / nu)))
            ys.append(m * (Lf ** (beta_exp / nu)))
            labels.append(L)
    return np.array(xs), np.array(ys), np.array(labels)

def prepare_collapse_chi(results_by_L, lambda_c_est, gamma_exp, nu):
    xs, ys, labels = [], [], []
    for L, items in results_by_L.items():
        Lf = float(L)
        for r in items:
            lam = r["lambda"]
            chi = r["susceptibility_mean"]
            xs.append((lam - lambda_c_est) * (Lf ** (1.0 / nu)))
            ys.append(chi / (Lf ** (gamma_exp / nu)))
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


# ============================================================
# 6. BOOTSTRAP
# ============================================================

def bootstrap_analysis(results_by_L, cfg):
    rng = np.random.default_rng(cfg.base_seed + 999999)

    binder_lambdas = []
    chi_lambdas = []
    order_nus = []
    order_betas = []
    chi_nus = []
    chi_gammas = []

    for b in range(cfg.bootstrap_samples):
        boot = bootstrap_resample_results(results_by_L, rng)

        lambda_c_binder, _ = estimate_lambda_c_from_binder(boot)
        lambda_c_chi, _ = estimate_lambda_c_from_chi(boot)
        lambda_c_est = lambda_c_binder if lambda_c_binder is not None else lambda_c_chi
        if lambda_c_est is None:
            continue

        if lambda_c_binder is not None:
            binder_lambdas.append(lambda_c_binder)
        if lambda_c_chi is not None:
            chi_lambdas.append(lambda_c_chi)

        best_order = optimize_order_collapse(boot, lambda_c_est, cfg)
        best_chi = optimize_chi_collapse(boot, lambda_c_est, cfg)

        if best_order is not None:
            order_nus.append(best_order["nu"])
            order_betas.append(best_order["beta_exp"])

        if best_chi is not None:
            chi_nus.append(best_chi["nu"])
            chi_gammas.append(best_chi["gamma_exp"])

        if (b + 1) % 20 == 0:
            print(f"Bootstrap {b+1}/{cfg.bootstrap_samples}")

    def summarize(vals):
        vals = np.array(vals, dtype=float)
        if len(vals) == 0:
            return None
        return {
            "mean": float(np.mean(vals)),
            "std": float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0,
            "p16": float(np.percentile(vals, 16)),
            "p50": float(np.percentile(vals, 50)),
            "p84": float(np.percentile(vals, 84)),
            "n": int(len(vals)),
        }

    return {
        "lambda_c_binder": summarize(binder_lambdas),
        "lambda_c_chi": summarize(chi_lambdas),
        "nu_from_order_collapse": summarize(order_nus),
        "beta_exp_from_order_collapse": summarize(order_betas),
        "nu_from_chi_collapse": summarize(chi_nus),
        "gamma_exp_from_chi_collapse": summarize(chi_gammas),
    }


# ============================================================
# 7. PLOTS
# ============================================================

def plot_binder_crossing(results_by_L, output_prefix="criticality_advanced_2d"):
    plt.figure(figsize=(8, 6))
    for L in sorted(results_by_L.keys()):
        x = np.array([r["lambda"] for r in results_by_L[L]])
        y = np.array([r["binder_mean"] for r in results_by_L[L]])
        plt.plot(x, y, marker="o", label=f"L={L}")

    plt.title("Binder Cumulant Crossing")
    plt.xlabel("λ")
    plt.ylabel("U4")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{output_prefix}_binder_crossing.png", dpi=220)
    plt.close()

def plot_order_collapse(results_by_L, lambda_c_est, best_order, output_prefix="criticality_advanced_2d"):
    x, y, labels = prepare_collapse_order(results_by_L, lambda_c_est, best_order["beta_exp"], best_order["nu"])

    plt.figure(figsize=(8, 6))
    for L in sorted(np.unique(labels)):
        mask = labels == L
        plt.scatter(x[mask], y[mask], s=24, label=f"L={L}")

    plt.title(
        f"Order Parameter Collapse\n"
        f"λc={lambda_c_est:.4f}, ν={best_order['nu']:.4f}, β={best_order['beta_exp']:.4f}"
    )
    plt.xlabel(r"$(\lambda-\lambda_c)L^{1/\nu}$")
    plt.ylabel(r"$\langle |\Psi| \rangle L^{\beta/\nu}$")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{output_prefix}_order_collapse.png", dpi=220)
    plt.close()

def plot_chi_collapse(results_by_L, lambda_c_est, best_chi, output_prefix="criticality_advanced_2d"):
    x, y, labels = prepare_collapse_chi(results_by_L, lambda_c_est, best_chi["gamma_exp"], best_chi["nu"])

    plt.figure(figsize=(8, 6))
    for L in sorted(np.unique(labels)):
        mask = labels == L
        plt.scatter(x[mask], y[mask], s=24, label=f"L={L}")

    plt.title(
        f"Susceptibility Collapse\n"
        f"λc={lambda_c_est:.4f}, ν={best_chi['nu']:.4f}, γ={best_chi['gamma_exp']:.4f}"
    )
    plt.xlabel(r"$(\lambda-\lambda_c)L^{1/\nu}$")
    plt.ylabel(r"$\chi/L^{\gamma/\nu}$")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{output_prefix}_chi_collapse.png", dpi=220)
    plt.close()


# ============================================================
# 8. MAIN
# ============================================================

def main():
    cfg = CriticalityConfigAdvanced()
    exp = CoherenceFieldExperiment2DAdvanced(cfg)

    results_by_L = exp.scan_all_sizes()

    lambda_c_binder, crossings = estimate_lambda_c_from_binder(results_by_L)
    lambda_c_chi, peaks = estimate_lambda_c_from_chi(results_by_L)
    lambda_c_est = lambda_c_binder if lambda_c_binder is not None else lambda_c_chi

    best_order = None
    best_chi = None
    if lambda_c_est is not None:
        best_order = optimize_order_collapse(results_by_L, lambda_c_est, cfg)
        best_chi = optimize_chi_collapse(results_by_L, lambda_c_est, cfg)

    print("\nRunning bootstrap inference...")
    bootstrap = bootstrap_analysis(results_by_L, cfg)

    summary = {
        "config": asdict(cfg),
        "lambda_c_input": cfg.lambda_c,
        "lambda_c_est_from_binder": lambda_c_binder,
        "binder_crossings": crossings,
        "lambda_c_est_from_chi_peaks": lambda_c_chi,
        "chi_peaks": peaks,
        "lambda_c_est_final": lambda_c_est,
        "best_order_collapse": best_order,
        "best_chi_collapse": best_chi,
        "bootstrap": bootstrap,
        "results_by_L": results_by_L,
    }

    with open("criticality_advanced_2d_results.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    plot_binder_crossing(results_by_L)
    if lambda_c_est is not None and best_order is not None:
        plot_order_collapse(results_by_L, lambda_c_est, best_order)
    if lambda_c_est is not None and best_chi is not None:
        plot_chi_collapse(results_by_L, lambda_c_est, best_chi)

    print("\n=== 2D ADVANCED SUMMARY ===")
    print(f"Input λc: {cfg.lambda_c:.6f}")
    print(f"Binder λc estimate: {lambda_c_binder}")
    print(f"Chi-peak λc estimate: {lambda_c_chi}")
    print(f"Final λc estimate: {lambda_c_est}")

    print("\nBootstrap intervals:")
    for key, val in bootstrap.items():
        print(f"{key}: {val}")

    print("\nSaved files:")
    print(" - criticality_advanced_2d_results.json")
    print(" - criticality_advanced_2d_binder_crossing.png")
    print(" - criticality_advanced_2d_order_collapse.png")
    print(" - criticality_advanced_2d_chi_collapse.png")


if __name__ == "__main__":
    main()