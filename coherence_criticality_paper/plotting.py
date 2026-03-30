import numpy as np
import matplotlib.pyplot as plt
from analysis import prepare_collapse_order, prepare_collapse_chi

def plot_binder_crossing(results_by_L, path):
    plt.figure(figsize=(8, 6))
    for L in sorted(results_by_L.keys()):
        x = np.array([r["lambda"] for r in results_by_L[L]])
        y = np.array([r["binder_mean"] for r in results_by_L[L]])
        plt.plot(x, y, marker="o", label=f"L={L}")
    plt.xlabel("λ")
    plt.ylabel("U4")
    plt.title("Binder Cumulant Crossing")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=220)
    plt.close()

def plot_observables_grid(results_by_L, path):
    fig, axs = plt.subplots(2, 2, figsize=(12, 8))
    axs = axs.ravel()

    for L in sorted(results_by_L.keys()):
        x = np.array([r["lambda"] for r in results_by_L[L]])
        axs[0].plot(x, [r["order_parameter_mean"] for r in results_by_L[L]], marker="o", label=f"L={L}")
        axs[1].plot(x, [r["susceptibility_mean"] for r in results_by_L[L]], marker="o", label=f"L={L}")
        axs[2].plot(x, [r["binder_mean"] for r in results_by_L[L]], marker="o", label=f"L={L}")
        axs[3].plot(x, [r["correlation_length_mean"] for r in results_by_L[L]], marker="o", label=f"L={L}")

    titles = ["Order parameter", "Susceptibility", "Binder cumulant", "Correlation length"]
    ylabels = ["<|Ψ|>", "χ", "U4", "ξ"]

    for i, ax in enumerate(axs):
        ax.set_title(titles[i])
        ax.set_xlabel("λ")
        ax.set_ylabel(ylabels[i])
        ax.legend()

    plt.tight_layout()
    plt.savefig(path, dpi=220)
    plt.close()

def plot_order_collapse(results_by_L, lambda_c_est, best_order, path):
    x, y, labels = prepare_collapse_order(results_by_L, lambda_c_est, best_order["beta_exp"], best_order["nu"])
    plt.figure(figsize=(8, 6))
    for L in sorted(np.unique(labels)):
        mask = labels == L
        plt.scatter(x[mask], y[mask], s=22, label=f"L={L}")
    plt.xlabel(r"$(\lambda-\lambda_c)L^{1/\nu}$")
    plt.ylabel(r"$\langle |\Psi| \rangle L^{\beta/\nu}$")
    plt.title(f"Order Parameter Collapse\nλc={lambda_c_est:.4f}, ν={best_order['nu']:.4f}, β={best_order['beta_exp']:.4f}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=220)
    plt.close()

def plot_chi_collapse(results_by_L, lambda_c_est, best_chi, path):
    x, y, labels = prepare_collapse_chi(results_by_L, lambda_c_est, best_chi["gamma_exp"], best_chi["nu"])
    plt.figure(figsize=(8, 6))
    for L in sorted(np.unique(labels)):
        mask = labels == L
        plt.scatter(x[mask], y[mask], s=22, label=f"L={L}")
    plt.xlabel(r"$(\lambda-\lambda_c)L^{1/\nu}$")
    plt.ylabel(r"$\chi/L^{\gamma/\nu}$")
    plt.title(f"Susceptibility Collapse\nλc={lambda_c_est:.4f}, ν={best_chi['nu']:.4f}, γ={best_chi['gamma_exp']:.4f}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=220)
    plt.close()