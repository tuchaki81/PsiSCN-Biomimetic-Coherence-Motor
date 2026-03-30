import math
import numpy as np

class CoherenceField2D:
    def __init__(self, cfg):
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

        return psi + self.cfg.dtau * self.cfg.gamma * force + noise

    def energy_density(self, psi: np.ndarray, alpha: float) -> float:
        dx_field = (np.roll(psi, -1, axis=0) - psi) / self.cfg.dx
        dy_field = (np.roll(psi, -1, axis=1) - psi) / self.cfg.dx
        grad2 = np.mean(dx_field**2 + dy_field**2)
        return float(0.5 * grad2 + alpha * np.mean(psi**2) + self.cfg.beta * np.mean(psi**4))

    def k_eff(self, alpha: float) -> float:
        if alpha >= 0:
            return 0.0
        return (self.cfg.zeta / (4.0 * math.pi)) * (abs(alpha) / (2.0 * self.cfg.beta))

    def connected_correlation_radial(self, psi: np.ndarray) -> np.ndarray:
        L = psi.shape[0]
        mean = np.mean(psi)
        phi = psi - mean

        fphi = np.fft.fftn(phi)
        corr2d = np.fft.ifftn(np.abs(fphi) ** 2).real / (L * L)

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
        return radial / counts

    def estimate_correlation_length(self, corr: np.ndarray) -> float:
        c0 = corr[0]
        if c0 <= 0:
            return 0.0
        norm = corr / c0
        target = math.exp(-1.0)
        for r in range(1, len(norm)):
            if norm[r] < target:
                return float(r * self.cfg.dx)
        return float((len(norm) - 1) * self.cfg.dx)

    def binder_cumulant(self, m_samples: np.ndarray) -> float:
        m2 = np.mean(m_samples ** 2)
        m4 = np.mean(m_samples ** 4)
        if m2 <= 1e-14:
            return 0.0
        return 1.0 - m4 / (3.0 * (m2 ** 2))

    def integrated_autocorrelation_time(self, series: np.ndarray) -> float:
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
        x = np.asarray(series, dtype=float)
        n = len(x)
        if n < 2:
            return 0.0
        tau_int = self.integrated_autocorrelation_time(x)
        n_eff = max(1.0, n / (2.0 * tau_int))
        return float(np.std(x, ddof=1) / math.sqrt(n_eff))