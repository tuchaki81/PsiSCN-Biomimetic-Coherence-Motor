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
    L: int = 64                   # lado do lattice quadrado (aumente para 128+ depois)
    dx: float = 1.0
    gamma: float = 0.08
    beta: float = 0.08
    a0: float = 1.0
    lambda_c: float = 1.0
    T0: float = 0.03
    zeta: float = 1.0

    # simulation
    burn_in: int = 1500
    sample_steps: int = 2500
    sample_every: int = 10
    dtau: float = 1.0

    # scan
    lambda_min: float = 0.80
    lambda_max: float = 1.20
    lambda_points: int = 15       # reduzido para velocidade

    # reproducibility
    seed: int = 42

# ============================================================
# 2. TDGL 2D
# ============================================================

class CoherenceFieldExperiment2D:
    def __init__(self, cfg: CriticalityConfig):
        self.cfg = cfg
        self.rng = np.random.default_rng(cfg.seed)

    def alpha_of_lambda(self, lam: float) -> float:
        return self.cfg.a0 * (self.cfg.lambda_c - lam)

    def laplacian_periodic_2d(self, psi: np.ndarray) -> np.ndarray:
        """Laplaciano periódico em 2D"""
        lap_x = (np.roll(psi, -1, axis=0) + np.roll(psi, 1, axis=0) - 2.0 * psi)
        lap_y = (np.roll(psi, -1, axis=1) + np.roll(psi, 1, axis=1) - 2.0 * psi)
        return (lap_x + lap_y) / (self.cfg.dx ** 2)

    def step(self, psi: np.ndarray, alpha: float) -> np.ndarray:
        lap = self.laplacian_periodic_2d(psi)
        force = lap - 2.0 * alpha * psi - 4.0 * self.cfg.beta * (psi ** 3)
        noise = self.rng.normal(0.0, math.sqrt(2.0 * self.cfg.gamma * self.cfg.T0), size=psi.shape)
        psi = psi + self.cfg.dtau * self.cfg.gamma * force + noise
        return psi

    def initialize_field(self) -> np.ndarray:
        return self.rng.normal(0.0, 1e-3, size=(self.cfg.L, self.cfg.L))

    def connected_correlation(self, psi: np.ndarray) -> np.ndarray:
        """C(r) média nas direções x e y"""
        mean = psi.mean()
        corr_x = np.zeros(self.cfg.L)
        corr_y = np.zeros(self.cfg.L)
        for r in range(self.cfg.L):
            shifted_x = np.roll(psi, -r, axis=0)
            corr_x[r] = np.mean(psi * shifted_x) - mean**2
            shifted_y = np.roll(psi, -r, axis=1)
            corr_y[r] = np.mean(psi * shifted_y) - mean**2
        return (corr_x + corr_y) / 2.0

    def estimate_correlation_length(self, corr: np.ndarray) -> float:
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
        if alpha >= 0:
            return 0.0
        return (self.cfg.zeta / (4.0 * math.pi)) * (abs(alpha) / (2.0 * self.cfg.beta))

    def run_at_lambda(self, lam: float) -> dict:
        alpha = self.alpha_of_lambda(lam)
        psi = self.initialize_field()

        # burn-in
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
                m = np.mean(psi)
                magnetizations.append(m)
                abs_magnetizations.append(abs(m))

                # energia com gradiente 2D
                dx = (np.roll(psi, -1, axis=0) - psi) / self.cfg.dx
                dy = (np.roll(psi, -1, axis=1) - psi) / self.cfg.dx
                grad2 = np.mean(dx**2 + dy**2)
                energy_density = 0.5 * grad2 + alpha * np.mean(psi**2) + self.cfg.beta * np.mean(psi**4)
                energies.append(energy_density)

                corr_accum += self.connected_correlation(psi)
                corr_count += 1

        magnetizations = np.array(magnetizations)
        abs_magnetizations = np.array(abs_magnetizations)
        energies = np.array(energies)
        corr_mean = corr_accum / max(corr_count, 1)

        return {
            "lambda": lam,
            "alpha": alpha,
            "order_parameter": float(np.mean(abs_magnetizations)),
            "susceptibility": float(self.cfg.L**2 * np.var(magnetizations) / max(self.cfg.T0, 1e-12)),
            "binder": float(self.binder_cumulant(magnetizations)),
            "correlation_length": float(self.estimate_correlation_length(corr_mean)),
            "energy_density": float(np.mean(energies)),
            "k_eff": float(self.k_eff(alpha)),
            "corr": corr_mean.tolist(),
        }

    def plot_field_evolution(self, output_prefix="criticality"):
        """Heatmaps da evolução do campo Ψ — o mais bonito!"""
        lam_c = self.cfg.lambda_c
        alpha = self.alpha_of_lambda(lam_c)
        psi = self.initialize_field()

        snapshots = []
        titles = []

        # após burn-in
        for _ in range(self.cfg.burn_in):
            psi = self.step(psi, alpha)
        snapshots.append(psi.copy())
        titles.append("Após burn-in (desordenado)")

        # evolução perto da criticalidade
        for t in range(800):
            psi = self.step(psi, alpha)
            if t % 200 == 0:
                snapshots.append(psi.copy())
                titles.append(f"Evolução t={t} (λ={lam_c})")

        # fase coerente
        for _ in range(400):
            psi = self.step(psi, alpha)
        snapshots.append(psi.copy())
        titles.append("Fase coerente (domínios ordenados)")

        fig, axs = plt.subplots(1, len(snapshots), figsize=(20, 5))
        for ax, snap, title in zip(axs, snapshots, titles):
            im = ax.imshow(snap, cmap='RdBu_r', vmin=-1.5, vmax=1.5)
            ax.set_title(title, fontsize=11)
            ax.axis('off')
            fig.colorbar(im, ax=ax, fraction=0.046)
        plt.suptitle("ΨSCN 2D — Ignção espontânea de coerência (λ = λc)", fontsize=16)
        plt.tight_layout()
        plt.savefig(f"{output_prefix}_field_evolution_2d.png", dpi=220)
        plt.close()
        print(f"✅ Heatmap salvo: {output_prefix}_field_evolution_2d.png")

# ============================================================
# 3. ANÁLISE E PLOTS (reaproveitados)
# ============================================================

def plot_results_2d(results, cfg: CriticalityConfig, output_prefix="criticality"):
    # (mesmo plot 1D, só muda o título)
    lambdas = np.array([r["lambda"] for r in results])
    order = np.array([r["order_parameter"] for r in results])
    chi = np.array([r["susceptibility"] for r in results])
    binder = np.array([r["binder"] for r in results])
    xi = np.array([r["correlation_length"] for r in results])
    k = np.array([r["k_eff"] for r in results])

    fig, axs = plt.subplots(2, 3, figsize=(15, 8))
    axs = axs.ravel()
    for i, (data, title) in enumerate(zip([order, chi, binder, xi, k], 
                                         ["Order parameter <|Ψ|>", "Susceptibility χ", 
                                          "Binder cumulant U4", "Correlation length ξ", 
                                          "Emergent k_eff"])):
        axs[i].plot(lambdas, data, marker="o")
        axs[i].axvline(cfg.lambda_c, ls="--", alpha=0.6)
        axs[i].set_title(title)
        axs[i].set_xlabel("λ")
        axs[i].set_ylabel(title.split()[0])

    # correlation example
    idx_mid = len(results) // 2
    corr = np.array(results[idx_mid]["corr"])
    r = np.arange(len(corr)//2)
    axs[5].plot(r, corr[:len(r)], label=f"near-critical λ={results[idx_mid]['lambda']:.3f}")
    axs[5].set_title("Connected correlation C(r) (2D average)")
    axs[5].set_xlabel("r")
    axs[5].set_ylabel("C(r)")
    axs[5].legend()

    plt.tight_layout()
    plt.savefig(f"{output_prefix}_observables_2d.png", dpi=200)
    plt.close()

def estimate_lambda_c_from_susceptibility(results):
    chis = np.array([r["susceptibility"] for r in results])
    lambdas = np.array([r["lambda"] for r in results])
    idx = int(np.argmax(chis))
    return float(lambdas[idx]), float(chis[idx])

# ============================================================
# 4. MAIN 2D
# ============================================================

def main():
    cfg = CriticalityConfig()
    exp = CoherenceFieldExperiment2D(cfg)
    results = exp.scan()

    plot_results_2d(results, cfg)
    exp.plot_field_evolution()

    lambda_c_est, _ = estimate_lambda_c_from_susceptibility(results)
    print("\n=== ΨSCN 2D SUMMARY ===")
    print(f"Input λc: {cfg.lambda_c:.6f}")
    print(f"Estimated λc from χ peak: {lambda_c_est:.6f}")
    print("Arquivos salvos:")
    print("   • criticality_observables_2d.png")
    print("   • criticality_field_evolution_2d.png  ← veja a ignição!")

if __name__ == "__main__":
    main()