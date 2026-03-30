import math
import json
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass, asdict


# ============================================================
# 1. CONFIG
# ============================================================

@dataclass
class CriticalityConfig:
    # lattice
    L: int = 256                  # number of sites
    dx: float = 1.0               # lattice spacing

    # dynamics
    gamma: float = 0.08           # relaxation rate
    beta: float = 0.08            # quartic coupling (>0)
    a0: float = 1.0               # alpha(λ)=a0*(λc-λ)
    lambda_c: float = 1.0         # nominal critical point
    T0: float = 0.03              # noise amplitude
    zeta: float = 1.0             # geometric/gravitational coupling

    # simulation
    burn_in: int = 4000
    sample_steps: int = 6000
    sample_every: int = 10
    dtau: float = 1.0             # absorbed into gamma convention

    # scan
    lambda_min: float = 0.80
    lambda_max: float = 1.20
    lambda_points: int = 31

    # reproducibility
    seed: int = 42


# ============================================================
# 2. TDGL / MODEL A DYNAMICS
# ============================================================

class CoherenceFieldExperiment:
    def __init__(self, cfg: CriticalityConfig):
        self.cfg = cfg
        self.rng = np.random.default_rng(cfg.seed)

    def alpha_of_lambda(self, lam: float) -> float:
        # positive below critical point, negative above
        return self.cfg.a0 * (self.cfg.lambda_c - lam)

    def laplacian_periodic(self, psi: np.ndarray) -> np.ndarray:
        # 1D periodic lattice
        return (np.roll(psi, -1) - 2.0 * psi + np.roll(psi, 1)) / (self.cfg.dx ** 2)

    def step(self, psi: np.ndarray, alpha: float) -> np.ndarray:
        """
        TDGL update:
        dψ/dτ = γ [ ∇²ψ - 2α ψ - 4β ψ^3 ] + η
        """
        lap = self.laplacian_periodic(psi)
        force = lap - 2.0 * alpha * psi - 4.0 * self.cfg.beta * (psi ** 3)

        # Gaussian white noise
        noise = self.rng.normal(
            loc=0.0,
            scale=math.sqrt(2.0 * self.cfg.gamma * self.cfg.T0),
            size=psi.shape
        )

        psi = psi + self.cfg.dtau * self.cfg.gamma * force + noise
        return psi

    def initialize_field(self) -> np.ndarray:
        # small random fluctuations around zero
        return self.rng.normal(0.0, 1e-3, size=(self.cfg.L,))

    def connected_correlation(self, psi: np.ndarray) -> np.ndarray:
        """
        C(r) = <ψ(x)ψ(x+r)> - <ψ>^2
        estimated from one configuration via translational averaging.
        """
        mean = psi.mean()
        corr = np.zeros(self.cfg.L)
        for r in range(self.cfg.L):
            corr[r] = np.mean(psi * np.roll(psi, -r)) - mean**2
        return corr

    def estimate_correlation_length(self, corr: np.ndarray) -> float:
        """
        Simple exponential proxy:
        find first positive r where C(r)/C(0) < exp(-1).
        If not found, return large finite proxy.
        """
        c0 = corr[0]
        if c0 <= 0:
            return 0.0

        norm = corr / c0
        target = math.exp(-1.0)

        for r in range(1, len(norm)//2):
            if norm[r] < target:
                return float(r * self.cfg.dx)

        return float((len(norm)//2) * self.cfg.dx)

    def binder_cumulant(self, m_samples: np.ndarray) -> float:
        m2 = np.mean(m_samples ** 2)
        m4 = np.mean(m_samples ** 4)
        if m2 <= 1e-14:
            return 0.0
        return 1.0 - m4 / (3.0 * (m2 ** 2))

    def k_eff(self, alpha: float) -> float:
        """
        Derived only in the ordered phase, from:
        <ψ>^2 = |α|/(2β)
        k = (ζ/4π) * |α|/(2β)
        """
        if alpha >= 0:
            return 0.0
        return (self.cfg.zeta / (4.0 * math.pi)) * (abs(alpha) / (2.0 * self.cfg.beta))

    def run_at_lambda(self, lam: float) -> dict:
        alpha = self.alpha_of_lambda(lam)
        psi = self.initialize_field()

        # burn-in
        for _ in range(self.cfg.burn_in):
            psi = self.step(psi, alpha)

        # sampling
        magnetizations = []
        abs_magnetizations = []
        energies = []
        corr_accum = np.zeros(self.cfg.L)
        corr_count = 0

        for t in range(self.cfg.sample_steps):
            psi = self.step(psi, alpha)

            if t % self.cfg.sample_every == 0:
                m = np.mean(psi)
                magnetizations.append(m)
                abs_magnetizations.append(abs(m))

                # crude free-energy density proxy
                grad = (np.roll(psi, -1) - psi) / self.cfg.dx
                energy_density = 0.5 * np.mean(grad**2) + alpha * np.mean(psi**2) + self.cfg.beta * np.mean(psi**4)
                energies.append(energy_density)

                corr_accum += self.connected_correlation(psi)
                corr_count += 1

        magnetizations = np.array(magnetizations)
        abs_magnetizations = np.array(abs_magnetizations)
        energies = np.array(energies)
        corr_mean = corr_accum / max(corr_count, 1)

        # observables
        order_parameter = float(np.mean(abs_magnetizations))

        # susceptibility from fluctuations of magnetization
        chi = float(self.cfg.L * np.var(magnetizations) / max(self.cfg.T0, 1e-12))

        binder = float(self.binder_cumulant(magnetizations))
        xi = float(self.estimate_correlation_length(corr_mean))
        k_val = float(self.k_eff(alpha))

        return {
            "lambda": lam,
            "alpha": alpha,
            "order_parameter": order_parameter,
            "susceptibility": chi,
            "binder": binder,
            "correlation_length": xi,
            "energy_density": float(np.mean(energies)),
            "k_eff": k_val,
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


# ============================================================
# 3. ANALYSIS
# ============================================================

def estimate_lambda_c_from_susceptibility(results):
    chis = np.array([r["susceptibility"] for r in results])
    lambdas = np.array([r["lambda"] for r in results])
    idx = int(np.argmax(chis))
    return float(lambdas[idx]), float(chis[idx])

def estimate_beta_exponent(results, lambda_c_est, fit_points=6):
    """
    Fit log(m) ~ beta_exp * log(λ - λc) on the ordered side.
    """
    ordered = []
    for r in results:
        dl = r["lambda"] - lambda_c_est
        m = r["order_parameter"]
        if dl > 0 and m > 1e-8:
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
    Fit log(ξ) ~ -nu * log|λ - λc|
    using both sides near criticality.
    """
    pts = []
    for r in results:
        dl = abs(r["lambda"] - lambda_c_est)
        xi = r["correlation_length"]
        if dl > 1e-8 and xi > 1e-8:
            pts.append((dl, xi))

    pts = sorted(pts, key=lambda z: z[0])[:max(fit_points, 4)]
    if len(pts) < 4:
        return None

    x = np.log(np.array([p[0] for p in pts]))
    y = np.log(np.array([p[1] for p in pts]))
    slope, intercept = np.polyfit(x, y, 1)
    return float(-slope), float(intercept)


# ============================================================
# 4. PLOTS
# ============================================================

def plot_results(results, cfg: CriticalityConfig, output_prefix="criticality"):
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
    axs[0].set_title("Order parameter  <|Ψ|>")
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

    # show correlation function for three representative points
    idx_low = 0
    idx_mid = len(results) // 2
    idx_high = len(results) - 1
    for idx, label in [(idx_low, "subcritical"), (idx_mid, "near-critical"), (idx_high, "supercritical")]:
        corr = np.array(results[idx]["corr"])
        r = np.arange(len(corr)//2)
        axs[5].plot(r, corr[:len(r)], label=f"{label} λ={results[idx]['lambda']:.3f}")
    axs[5].set_title("Connected correlation C(r)")
    axs[5].set_xlabel("r")
    axs[5].set_ylabel("C(r)")
    axs[5].legend()

    plt.tight_layout()
    plt.savefig(f"{output_prefix}_observables.png", dpi=200)
    plt.close()


# ============================================================
# 5. MAIN
# ============================================================

def main():
    cfg = CriticalityConfig()
    exp = CoherenceFieldExperiment(cfg)
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

    with open("criticality_results.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    plot_results(results, cfg, output_prefix="criticality")

    print("\n=== SUMMARY ===")
    print(f"Input λc: {cfg.lambda_c:.6f}")
    print(f"Estimated λc from χ peak: {lambda_c_est:.6f}")
    if beta_fit is not None:
        print(f"Estimated beta exponent: {beta_fit[0]:.4f}")
    else:
        print("Estimated beta exponent: insufficient data")
    if nu_fit is not None:
        print(f"Estimated nu exponent: {nu_fit[0]:.4f}")
    else:
        print("Estimated nu exponent: insufficient data")
    print("Saved: criticality_results.json")
    print("Saved: criticality_observables.png")


if __name__ == "__main__":
    main()