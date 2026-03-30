import math
import json
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass, asdict


# ============================================================
# 1. CONFIG (2D)
# ============================================================

@dataclass
class CriticalityConfig:
    # lattice
    L: int = 64                   # side of square lattice
    dx: float = 1.0

    # TDGL dynamics
    gamma: float = 0.08           # relaxation rate
    beta: float = 0.08            # quartic coupling (>0)
    a0: float = 1.0               # alpha(lambda)=a0*(lambda_c-lambda)
    lambda_c: float = 1.0         # nominal critical point
    T0: float = 0.03              # effective noise temperature
    zeta: float = 1.0             # geometric/gravitational coupling

    # simulation
    burn_in: int = 2000
    sample_steps: int = 3000
    sample_every: int = 10
    dtau: float = 1.0

    # scan
    lambda_min: float = 0.80
    lambda_max: float = 1.20
    lambda_points: int = 17

    # snapshots around criticality
    snapshot_offset: float = 0.08
    snapshot_steps_sub: int = 1200
    snapshot_steps_crit: int = 1200
    snapshot_steps_sup: int = 1200

    # reproducibility
    seed: int = 42


# ============================================================
# 2. TDGL 2D EXPERIMENT
# ============================================================

class CoherenceFieldExperiment2D:
    def __init__(self, cfg: CriticalityConfig):
        self.cfg = cfg
        self.rng = np.random.default_rng(cfg.seed)

    def alpha_of_lambda(self, lam: float) -> float:
        return self.cfg.a0 * (self.cfg.lambda_c - lam)

    def initialize_field(self) -> np.ndarray:
        return self.rng.normal(0.0, 1e-3, size=(self.cfg.L, self.cfg.L))

    def laplacian_periodic_2d(self, psi: np.ndarray) -> np.ndarray:
        lap_x = np.roll(psi, -1, axis=0) + np.roll(psi, 1, axis=0) - 2.0 * psi
        lap_y = np.roll(psi, -1, axis=1) + np.roll(psi, 1, axis=1) - 2.0 * psi
        return (lap_x + lap_y) / (self.cfg.dx ** 2)

    def step(self, psi: np.ndarray, alpha: float) -> np.ndarray:
        """
        TDGL / Model-A:
        dPsi/dtau = gamma [ ∇²Psi - 2 alpha Psi - 4 beta Psi^3 ] + eta
        """
        lap = self.laplacian_periodic_2d(psi)
        force = lap - 2.0 * alpha * psi - 4.0 * self.cfg.beta * (psi ** 3)

        noise = self.rng.normal(
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
        """
        Emergent effective coupling:
        k = (zeta / 4pi) * |alpha| / (2 beta), only in supercritical regime.
        """
        if alpha >= 0:
            return 0.0
        return (self.cfg.zeta / (4.0 * math.pi)) * (abs(alpha) / (2.0 * self.cfg.beta))

    def connected_correlation_axis_average(self, psi: np.ndarray) -> np.ndarray:
        """
        Simple directional average in x and y.
        This is not fully isotropic radial averaging, but is a useful first proxy.
        """
        mean = psi.mean()
        corr_x = np.zeros(self.cfg.L)
        corr_y = np.zeros(self.cfg.L)

        for r in range(self.cfg.L):
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

    def equilibrate(self, lam: float, extra_steps: int = 0) -> np.ndarray:
        alpha = self.alpha_of_lambda(lam)
        psi = self.initialize_field()

        for _ in range(self.cfg.burn_in + extra_steps):
            psi = self.step(psi, alpha)

        return psi

    def run_at_lambda(self, lam: float) -> dict:
        alpha = self.alpha_of_lambda(lam)
        psi = self.initialize_field()

        # Burn-in
        for _ in range(self.cfg.burn_in):
            psi = self.step(psi, alpha)

        magnetizations = []
        abs_magnetizations = []
        energies = []
        corr_accum = np.zeros(self.cfg.L)
        corr_count = 0

        for t in range(self.cfg.sample_steps):
            psi = self.step(psi, alpha)

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

        susceptibility = float((self.cfg.L ** 2) * np.var(magnetizations) / max(self.cfg.T0, 1e-12))

        return {
            "lambda": float(lam),
            "alpha": float(alpha),
            "order_parameter": float(np.mean(abs_magnetizations)),
            "susceptibility": susceptibility,
            "binder": float(self.binder_cumulant(magnetizations)),
            "correlation_length": float(self.estimate_correlation_length(corr_mean)),
            "energy_density": float(np.mean(energies)),
            "k_eff": float(self.k_eff(alpha)),
            "corr": corr_mean.tolist(),
        }

    def scan(self):
        lambdas = np.linspace(self.cfg.lambda_min, self.cfg.lambda_max, self.cfg.lambda_points)
        results = []

        for lam in lambdas:
            res = self.run_at_lambda(float(lam))
            results.append(res)
            print(
                f"λ={res['lambda']:.4f} | α={res['alpha']:.4f} | "
                f"m={res['order_parameter']:.5f} | χ={res['susceptibility']:.5f} | "
                f"U4={res['binder']:.5f} | ξ={res['correlation_length']:.3f} | "
                f"k={res['k_eff']:.5f}"
            )

        return results

    def plot_regime_snapshots(self, output_prefix="criticality_2d"):
        """
        Create separate snapshots for subcritical, critical, and supercritical regimes.
        """
        lam_sub = self.cfg.lambda_c - self.cfg.snapshot_offset
        lam_crit = self.cfg.lambda_c
        lam_sup = self.cfg.lambda_c + self.cfg.snapshot_offset

        regimes = [
            ("Subcritical", lam_sub, self.cfg.snapshot_steps_sub),
            ("Critical", lam_crit, self.cfg.snapshot_steps_crit),
            ("Supercritical", lam_sup, self.cfg.snapshot_steps_sup),
        ]

        snapshots = []
        meta = []

        for label, lam, extra_steps in regimes:
            alpha = self.alpha_of_lambda(lam)
            psi = self.equilibrate(lam=lam, extra_steps=extra_steps)
            snapshots.append(psi.copy())
            meta.append((label, lam, alpha))

        vmax = max(np.max(np.abs(s)) for s in snapshots)
        vmax = max(vmax, 1e-6)

        fig, axs = plt.subplots(1, 3, figsize=(15, 5))

        for ax, snap, (label, lam, alpha) in zip(axs, snapshots, meta):
            im = ax.imshow(snap, cmap="RdBu_r", vmin=-vmax, vmax=vmax)
            ax.set_title(f"{label}\nλ={lam:.3f}, α={alpha:.3f}")
            ax.axis("off")
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        plt.suptitle("2D Coherence Field Regimes", fontsize=16)
        plt.tight_layout()
        plt.savefig(f"{output_prefix}_regimes.png", dpi=220)
        plt.close()
        print(f"Saved: {output_prefix}_regimes.png")

    def plot_critical_evolution(self, output_prefix="criticality_2d"):
        """
        Show field evolution exactly at the nominal critical point.
        Useful to visualize large fluctuations and domain formation attempts.
        """
        lam = self.cfg.lambda_c
        alpha = self.alpha_of_lambda(lam)
        psi = self.initialize_field()

        snapshots = []
        steps_mark = []

        total_steps = self.cfg.burn_in + 1200
        marks = [0, self.cfg.burn_in // 2, self.cfg.burn_in, self.cfg.burn_in + 400, self.cfg.burn_in + 1200]

        for t in range(total_steps + 1):
            if t in marks:
                snapshots.append(psi.copy())
                steps_mark.append(t)

            if t < total_steps:
                psi = self.step(psi, alpha)

        vmax = max(np.max(np.abs(s)) for s in snapshots)
        vmax = max(vmax, 1e-6)

        fig, axs = plt.subplots(1, len(snapshots), figsize=(20, 4.5))
        if len(snapshots) == 1:
            axs = [axs]

        for ax, snap, step_mark in zip(axs, snapshots, steps_mark):
            im = ax.imshow(snap, cmap="RdBu_r", vmin=-vmax, vmax=vmax)
            ax.set_title(f"τ-step {step_mark}")
            ax.axis("off")
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        plt.suptitle(f"Critical Evolution at λ = λc = {lam:.3f}", fontsize=16)
        plt.tight_layout()
        plt.savefig(f"{output_prefix}_critical_evolution.png", dpi=220)
        plt.close()
        print(f"Saved: {output_prefix}_critical_evolution.png")


# ============================================================
# 3. ANALYSIS
# ============================================================

def estimate_lambda_c_from_susceptibility(results):
    chis = np.array([r["susceptibility"] for r in results])
    lambdas = np.array([r["lambda"] for r in results])
    idx = int(np.argmax(chis))
    return float(lambdas[idx]), float(chis[idx])

def estimate_beta_exponent(results, lambda_c_est, fit_points=5):
    """
    Fit log(m) ~ beta_exp * log(λ - λc_est) on ordered side.
    """
    ordered = []
    for r in results:
        dl = r["lambda"] - lambda_c_est
        m = r["order_parameter"]
        if dl > 0 and m > 1e-10:
            ordered.append((dl, m))

    if len(ordered) < 3:
        return None

    ordered = ordered[:fit_points]
    x = np.log(np.array([p[0] for p in ordered]))
    y = np.log(np.array([p[1] for p in ordered]))
    slope, intercept = np.polyfit(x, y, 1)
    return float(slope), float(intercept)

def estimate_nu_exponent(results, lambda_c_est, fit_points=6):
    """
    Fit log(ξ) ~ -nu * log|λ - λc_est|.
    """
    pts = []
    for r in results:
        dl = abs(r["lambda"] - lambda_c_est)
        xi = r["correlation_length"]
        if dl > 1e-10 and xi > 1e-10:
            pts.append((dl, xi))

    if len(pts) < 4:
        return None

    pts = sorted(pts, key=lambda z: z[0])[:fit_points]
    x = np.log(np.array([p[0] for p in pts]))
    y = np.log(np.array([p[1] for p in pts]))
    slope, intercept = np.polyfit(x, y, 1)
    return float(-slope), float(intercept)


# ============================================================
# 4. PLOTTING
# ============================================================

def plot_results_2d(results, cfg: CriticalityConfig, output_prefix="criticality_2d"):
    lambdas = np.array([r["lambda"] for r in results])
    order = np.array([r["order_parameter"] for r in results])
    chi = np.array([r["susceptibility"] for r in results])
    binder = np.array([r["binder"] for r in results])
    xi = np.array([r["correlation_length"] for r in results])
    k = np.array([r["k_eff"] for r in results])

    fig, axs = plt.subplots(2, 3, figsize=(15, 8))
    axs = axs.ravel()

    axs[0].plot(lambdas, order, marker="o")
    axs[0].axvline(cfg.lambda_c, ls="--", alpha=0.6)
    axs[0].set_title("Order parameter <|Ψ|>")
    axs[0].set_xlabel("λ")
    axs[0].set_ylabel("<|Ψ|>")

    axs[1].plot(lambdas, chi, marker="o")
    axs[1].axvline(cfg.lambda_c, ls="--", alpha=0.6)
    axs[1].set_title("Susceptibility χ")
    axs[1].set_xlabel("λ")
    axs[1].set_ylabel("χ")

    axs[2].plot(lambdas, binder, marker="o")
    axs[2].axvline(cfg.lambda_c, ls="--", alpha=0.6)
    axs[2].set_title("Binder cumulant U4")
    axs[2].set_xlabel("λ")
    axs[2].set_ylabel("U4")

    axs[3].plot(lambdas, xi, marker="o")
    axs[3].axvline(cfg.lambda_c, ls="--", alpha=0.6)
    axs[3].set_title("Correlation length proxy ξ")
    axs[3].set_xlabel("λ")
    axs[3].set_ylabel("ξ")

    axs[4].plot(lambdas, k, marker="o")
    axs[4].axvline(cfg.lambda_c, ls="--", alpha=0.6)
    axs[4].set_title("Emergent k_eff")
    axs[4].set_xlabel("λ")
    axs[4].set_ylabel("k_eff")

    idx_low = 0
    idx_mid = len(results) // 2
    idx_high = len(results) - 1

    for idx, label in [(idx_low, "subcritical"), (idx_mid, "near-critical"), (idx_high, "supercritical")]:
        corr = np.array(results[idx]["corr"])
        r = np.arange(len(corr) // 2)
        axs[5].plot(r, corr[:len(r)], label=f"{label} λ={results[idx]['lambda']:.3f}")

    axs[5].set_title("Connected correlation C(r)")
    axs[5].set_xlabel("r")
    axs[5].set_ylabel("C(r)")
    axs[5].legend()

    plt.tight_layout()
    plt.savefig(f"{output_prefix}_observables.png", dpi=220)
    plt.close()
    print(f"Saved: {output_prefix}_observables.png")


# ============================================================
# 5. MAIN
# ============================================================

def main():
    cfg = CriticalityConfig()
    exp = CoherenceFieldExperiment2D(cfg)

    results = exp.scan()

    lambda_c_est, chi_peak = estimate_lambda_c_from_susceptibility(results)
    beta_fit = estimate_beta_exponent(results, lambda_c_est)
    nu_fit = estimate_nu_exponent(results, lambda_c_est)

    summary = {
        "config": asdict(cfg),
        "lambda_c_input": cfg.lambda_c,
        "lambda_c_est_from_chi_peak": lambda_c_est,
        "chi_peak": chi_peak,
        "beta_exp_fit": beta_fit[0] if beta_fit is not None else None,
        "nu_exp_fit": nu_fit[0] if nu_fit is not None else None,
        "results": results,
    }

    with open("criticality_2d_results.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    plot_results_2d(results, cfg, output_prefix="criticality_2d")
    exp.plot_regime_snapshots(output_prefix="criticality_2d")
    exp.plot_critical_evolution(output_prefix="criticality_2d")

    print("\n=== 2D CRITICALITY SUMMARY ===")
    print(f"Input λc: {cfg.lambda_c:.6f}")
    print(f"Estimated λc from χ peak: {lambda_c_est:.6f}")
    print(f"χ peak: {chi_peak:.6f}")
    if beta_fit is not None:
        print(f"Estimated beta exponent: {beta_fit[0]:.4f}")
    else:
        print("Estimated beta exponent: insufficient data")
    if nu_fit is not None:
        print(f"Estimated nu exponent: {nu_fit[0]:.4f}")
    else:
        print("Estimated nu exponent: insufficient data")

    print("\nSaved files:")
    print(" - criticality_2d_results.json")
    print(" - criticality_2d_observables.png")
    print(" - criticality_2d_regimes.png")
    print(" - criticality_2d_critical_evolution.png")


if __name__ == "__main__":
    main()
