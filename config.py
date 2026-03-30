from dataclasses import dataclass, asdict

@dataclass
class CriticalityConfig:
    sizes: tuple = (24, 32, 48, 64)

    dx: float = 1.0

    gamma: float = 0.08
    beta: float = 0.08
    a0: float = 1.0
    lambda_c: float = 1.0
    T0: float = 0.03
    zeta: float = 1.0

    burn_in: int = 1600
    sample_steps: int = 2600
    sample_every: int = 10
    dtau: float = 1.0

    lambda_min: float = 0.90
    lambda_max: float = 1.10
    lambda_points: int = 15

    repeats: int = 6
    base_seed: int = 42

    bootstrap_samples: int = 100

    nu_min: float = 0.4
    nu_max: float = 1.4
    nu_points: int = 31

    beta_exp_min: float = 0.05
    beta_exp_max: float = 0.30
    beta_exp_points: int = 31

    gamma_exp_min: float = 0.5
    gamma_exp_max: float = 2.0
    gamma_exp_points: int = 31

    max_lag_fraction: float = 0.25
    autocorr_cutoff: float = 0.0

    def to_dict(self):
        return asdict(self)