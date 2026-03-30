import numpy as np
from pathlib import Path

from config import CriticalityConfig
from coherence_model import CoherenceField2D
from analysis import (
    estimate_lambda_c_from_binder,
    estimate_lambda_c_from_chi,
    optimize_order_collapse,
    optimize_chi_collapse,
)
from plotting import (
    plot_binder_crossing,
    plot_observables_grid,
    plot_order_collapse,
    plot_chi_collapse,
)
from io_utils import save_json

OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

def run_single(model, cfg, lam, L, seed):
    alpha = model.alpha_of_lambda(lam)
    rng = np.random.default_rng(seed)
    psi = model.initialize_field(L, rng)

    for _ in range(cfg.burn_in):
        psi = model.step(psi, alpha, rng)

    magnetizations = []
    abs_magnetizations = []
    energies = []
    corr_accum = None
    corr_count = 0

    for t in range(cfg.sample_steps):
        psi = model.step(psi, alpha, rng)

        if t % cfg.sample_every == 0:
            m = float(np.mean(psi))
            magnetizations.append(m)
            abs_magnetizations.append(abs(m))
            energies.append(model.energy_density(psi, alpha))

            corr = model.connected_correlation_radial(psi)
            if corr_accum is None:
                corr_accum = np.zeros_like(corr)
            corr_accum += corr
            corr_count += 1

    magnetizations = np.array(magnetizations)
    abs_magnetizations = np.array(abs_magnetizations)
    energies = np.array(energies)
    corr_mean = corr_accum / max(corr_count, 1)

    return {
        "order_parameter": float(np.mean(abs_magnetizations)),
        "order_parameter_se_timecorr": model.effective_standard_error(abs_magnetizations),
        "susceptibility": float((L ** 2) * np.var(magnetizations) / max(cfg.T0, 1e-12)),
        "binder": float(model.binder_cumulant(magnetizations)),
        "correlation_length": float(model.estimate_correlation_length(corr_mean)),
        "energy_density": float(np.mean(energies)),
        "energy_density_se_timecorr": model.effective_standard_error(energies),
        "k_eff": float(model.k_eff(alpha)),
        "tau_int_m": float(model.integrated_autocorrelation_time(magnetizations)),
        "tau_int_absm": float(model.integrated_autocorrelation_time(abs_magnetizations)),
    }

def run_ensemble(model, cfg, lam, L):
    ensemble = []
    for rep in range(cfg.repeats):
        seed = cfg.base_seed + 10000 * L + 100 * int(round(lam * 1000)) + rep
        ensemble.append(run_single(model, cfg, lam, L, seed))

    def mean_std(key):
        vals = np.array([e[key] for e in ensemble], dtype=float)
        return float(np.mean(vals)), float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0

    order_m, order_s = mean_std("order_parameter")
    chi_m, chi_s = mean_std("susceptibility")
    binder_m, binder_s = mean_std("binder")
    xi_m, xi_s = mean_std("correlation_length")
    e_m, e_s = mean_std("energy_density")
    k_m, k_s = mean_std("k_eff")

    return {
        "L": int(L),
        "lambda": float(lam),
        "alpha": float(model.alpha_of_lambda(lam)),
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
    }

def main():
    cfg = CriticalityConfig()
    model = CoherenceField2D(cfg)

    lambdas = np.linspace(cfg.lambda_min, cfg.lambda_max, cfg.lambda_points)
    results_by_L = {int(L): [] for L in cfg.sizes}

    for L in cfg.sizes:
        print(f"\n=== Scanning L={L} ===")
        for lam in lambdas:
            res = run_ensemble(model, cfg, float(lam), int(L))
            results_by_L[int(L)].append(res)
            print(
                f"L={L:3d} | λ={res['lambda']:.4f} | "
                f"m={res['order_parameter_mean']:.5f}±{res['order_parameter_std']:.5f} | "
                f"χ={res['susceptibility_mean']:.5f}±{res['susceptibility_std']:.5f} | "
                f"U4={res['binder_mean']:.5f}±{res['binder_std']:.5f}"
            )

    lambda_c_binder, crossings = estimate_lambda_c_from_binder(results_by_L)
    lambda_c_chi, peaks = estimate_lambda_c_from_chi(results_by_L)
    lambda_c_est = lambda_c_binder if lambda_c_binder is not None else lambda_c_chi

    best_order = optimize_order_collapse(results_by_L, lambda_c_est, cfg) if lambda_c_est is not None else None
    best_chi = optimize_chi_collapse(results_by_L, lambda_c_est, cfg) if lambda_c_est is not None else None

    raw = {
        "config": cfg.to_dict(),
        "results_by_L": results_by_L,
    }

    summary = {
        "config": cfg.to_dict(),
        "lambda_c_input": cfg.lambda_c,
        "lambda_c_est_from_binder": lambda_c_binder,
        "binder_crossings": crossings,
        "lambda_c_est_from_chi_peaks": lambda_c_chi,
        "chi_peaks": peaks,
        "lambda_c_est_final": lambda_c_est,
        "best_order_collapse": best_order,
        "best_chi_collapse": best_chi,
    }

    save_json(OUTPUT_DIR / "raw_results.json", raw)
    save_json(OUTPUT_DIR / "summary_results.json", summary)

    plot_binder_crossing(results_by_L, OUTPUT_DIR / "fig_binder_crossing.png")
    plot_observables_grid(results_by_L, OUTPUT_DIR / "fig_observables_grid.png")

    if lambda_c_est is not None and best_order is not None:
        plot_order_collapse(results_by_L, lambda_c_est, best_order, OUTPUT_DIR / "fig_order_collapse.png")

    if lambda_c_est is not None and best_chi is not None:
        plot_chi_collapse(results_by_L, lambda_c_est, best_chi, OUTPUT_DIR / "fig_susceptibility_collapse.png")

    print("\nSaved in outputs/")

if __name__ == "__main__":
    main()