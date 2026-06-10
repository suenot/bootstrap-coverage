"""Return-stream DGPs with analytically known annualized Sharpe ratios.

Five families of daily-return processes, each normalized so that the
*unconditional* mean is ``mu_daily`` and the *unconditional* standard deviation
is ``sigma_daily``. The target parameter throughout the study is the population
annualized Sharpe ratio

    SR_ann = sqrt(252) * mu_daily / sigma_daily,

which is known **exactly** for every family by construction:

* ``iid_gaussian``       -- r_t = mu + sigma * z_t, z_t ~ N(0,1).
* ``iid_student_t``      -- r_t = mu + sigma * t_nu / sqrt(nu/(nu-2)); the
                            Student-t is standardized to unit variance, so the
                            unconditional sd is exactly sigma (requires nu > 2;
                            we sample nu >= 5 so the 4th moment exists too).
* ``ar1``                -- r_t = mu + x_t with x_t = phi*x_{t-1} + e_t and
                            sd(e) = sigma*sqrt(1-phi^2); x_0 is drawn from the
                            stationary N(0, sigma^2) law, so the process is
                            *exactly* stationary (no burn-in needed). Models
                            autocorrelated PnL streams, e.g. from overlapping
                            positions.
* ``garch``              -- GARCH(1,1): r_t = mu + e_t, e_t = s_t*z_t,
                            s_t^2 = omega + alpha*e_{t-1}^2 + beta*s_{t-1}^2
                            with omega = sigma^2*(1-alpha-beta), so the
                            unconditional variance is analytically
                            omega/(1-alpha-beta) = sigma^2. Initialized at the
                            unconditional variance and burned in for
                            ``GARCH_BURN_IN`` steps.
* ``regime_switching``   -- 2-state Markov volatility: r_t = mu + sig_{S_t}*z_t.
                            The state-dependent vols (sig_low, sig_high) are
                            solved from the stationary distribution of the
                            chain so that the unconditional variance is exactly
                            sigma^2; S_0 is drawn from the stationary
                            distribution (exactly stationary).

Serial dependence changes the *sampling distribution* of the Sharpe estimator
(precisely what the paper measures via CI coverage) but not the target
parameter: the plug-in Sharpe statistic remains consistent for SR_ann under
every family above.

Every parameter of every experiment is sampled by :func:`sample_config` and
recorded in the (frozen) :class:`DGPConfig` -- there are no hidden per-draw
constants. The module-level ``*_RANGE`` / ``*_CHOICES`` tuples below define the
sampling *distributions* and are themselves recorded in ``results.json`` meta.

Everything is deterministic given a seeded ``numpy.random.Generator``.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import numpy as np
from scipy.signal import lfilter

TRADING_DAYS: int = 252
GARCH_BURN_IN: int = 500  # burn-in steps discarded for GARCH (documented, fixed)

FAMILIES: tuple[str, ...] = (
    "iid_gaussian",
    "iid_student_t",
    "ar1",
    "garch",
    "regime_switching",
)
IID_FAMILIES: tuple[str, ...] = ("iid_gaussian", "iid_student_t")
DEPENDENT_FAMILIES: tuple[str, ...] = ("ar1", "garch", "regime_switching")

T_GRID: tuple[int, ...] = (250, 1000, 4000)

# ---- sampling distributions used by sample_config (recorded in results meta) --
SHARPE_ANNUAL_RANGE: tuple[float, float] = (0.0, 2.0)     # true annualized Sharpe
SIGMA_DAILY_RANGE: tuple[float, float] = (0.005, 0.02)    # ~8%..32% annualized vol
NU_CHOICES: tuple[float, ...] = (5.0, 6.0, 8.0, 12.0, 20.0)  # Student-t dof (>4)
PHI_CHOICES: tuple[float, ...] = (0.05, 0.10, 0.20, 0.30)    # AR(1) coefficient
GARCH_ALPHA_RANGE: tuple[float, float] = (0.03, 0.12)
GARCH_PERSISTENCE_RANGE: tuple[float, float] = (0.85, 0.98)  # alpha + beta
VOL_RATIO_RANGE: tuple[float, float] = (2.0, 4.0)            # sig_high / sig_low
P_STAY_LOW_RANGE: tuple[float, float] = (0.95, 0.99)
P_STAY_HIGH_RANGE: tuple[float, float] = (0.90, 0.98)


def sampling_ranges() -> dict:
    """The exact sampling distributions used by :func:`sample_config` (for meta)."""
    return {
        "sharpe_annual_range": SHARPE_ANNUAL_RANGE,
        "sigma_daily_range": SIGMA_DAILY_RANGE,
        "nu_choices": NU_CHOICES,
        "phi_choices": PHI_CHOICES,
        "garch_alpha_range": GARCH_ALPHA_RANGE,
        "garch_persistence_range": GARCH_PERSISTENCE_RANGE,
        "vol_ratio_range": VOL_RATIO_RANGE,
        "p_stay_low_range": P_STAY_LOW_RANGE,
        "p_stay_high_range": P_STAY_HIGH_RANGE,
        "garch_burn_in": GARCH_BURN_IN,
        "trading_days": TRADING_DAYS,
    }


# --------------------------------------------------------------------------- #
# config
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class DGPConfig:
    """Fully-specified ground-truth DGP for one experiment (all params recorded)."""

    family: str
    t_steps: int
    mu_daily: float
    sigma_daily: float
    # family-specific parameters (NaN when not applicable -- still recorded)
    nu: float = float("nan")             # Student-t degrees of freedom
    phi: float = float("nan")            # AR(1) coefficient
    garch_alpha: float = float("nan")
    garch_beta: float = float("nan")
    p_stay_low: float = float("nan")     # regime: P(stay | low-vol state)
    p_stay_high: float = float("nan")    # regime: P(stay | high-vol state)
    vol_ratio: float = float("nan")      # regime: sig_high / sig_low
    label: str = "custom"

    @property
    def true_sharpe_annual(self) -> float:
        """The exact population annualized Sharpe ratio (the coverage target)."""
        return math.sqrt(TRADING_DAYS) * self.mu_daily / self.sigma_daily

    @property
    def garch_persistence(self) -> float:
        return self.garch_alpha + self.garch_beta

    def as_record(self) -> dict:
        rec = {f"cfg_{k}": v for k, v in asdict(self).items()}
        rec["true_sharpe"] = self.true_sharpe_annual
        return rec


# --------------------------------------------------------------------------- #
# analytic helpers (regime-switching)
# --------------------------------------------------------------------------- #
def regime_stationary_probs(p_stay_low: float, p_stay_high: float) -> tuple[float, float]:
    """Stationary distribution (pi_low, pi_high) of the 2-state Markov chain."""
    q_low = 1.0 - p_stay_low    # P(low -> high)
    q_high = 1.0 - p_stay_high  # P(high -> low)
    pi_low = q_high / (q_low + q_high)
    return pi_low, 1.0 - pi_low


def regime_sigmas(cfg: DGPConfig) -> tuple[float, float]:
    """State vols (sig_low, sig_high) solving the unconditional-variance identity

    pi_low*sig_low^2 + pi_high*sig_high^2 = sigma_daily^2,  sig_high = ratio*sig_low.
    """
    pi_low, pi_high = regime_stationary_probs(cfg.p_stay_low, cfg.p_stay_high)
    sig_low = cfg.sigma_daily / math.sqrt(pi_low + pi_high * cfg.vol_ratio**2)
    return sig_low, cfg.vol_ratio * sig_low


# --------------------------------------------------------------------------- #
# generators (all vectorized across paths; n_paths=1 for single experiments)
# --------------------------------------------------------------------------- #
def _generate_iid_gaussian(cfg: DGPConfig, rng: np.random.Generator, n_paths: int) -> np.ndarray:
    z = rng.standard_normal((n_paths, cfg.t_steps))
    return cfg.mu_daily + cfg.sigma_daily * z


def _generate_iid_student_t(cfg: DGPConfig, rng: np.random.Generator, n_paths: int) -> np.ndarray:
    scale = math.sqrt(cfg.nu / (cfg.nu - 2.0))  # sd of raw t_nu
    t = rng.standard_t(cfg.nu, size=(n_paths, cfg.t_steps)) / scale
    return cfg.mu_daily + cfg.sigma_daily * t


def _generate_ar1(cfg: DGPConfig, rng: np.random.Generator, n_paths: int) -> np.ndarray:
    phi = cfg.phi
    innov_sd = cfg.sigma_daily * math.sqrt(1.0 - phi**2)
    e = rng.standard_normal((n_paths, cfg.t_steps)) * innov_sd
    x0 = rng.standard_normal(n_paths) * cfg.sigma_daily  # stationary start
    zi = (phi * x0)[:, None]
    x, _ = lfilter([1.0], [1.0, -phi], e, axis=1, zi=zi)
    return cfg.mu_daily + x


def _generate_garch(cfg: DGPConfig, rng: np.random.Generator, n_paths: int) -> np.ndarray:
    alpha, beta = cfg.garch_alpha, cfg.garch_beta
    if alpha + beta >= 1.0:
        raise ValueError(f"GARCH not covariance-stationary: alpha+beta={alpha + beta}")
    var_uncond = cfg.sigma_daily**2
    omega = var_uncond * (1.0 - alpha - beta)  # analytic unconditional variance
    total = cfg.t_steps + GARCH_BURN_IN
    z = rng.standard_normal((n_paths, total))
    out = np.empty((n_paths, total))
    sig2 = np.full(n_paths, var_uncond)
    for j in range(total):
        e = np.sqrt(sig2) * z[:, j]
        out[:, j] = e
        sig2 = omega + alpha * e * e + beta * sig2
    return cfg.mu_daily + out[:, GARCH_BURN_IN:]


def _generate_regime(cfg: DGPConfig, rng: np.random.Generator, n_paths: int) -> np.ndarray:
    pi_low, pi_high = regime_stationary_probs(cfg.p_stay_low, cfg.p_stay_high)
    sig_low, sig_high = regime_sigmas(cfg)
    t = cfg.t_steps
    z = rng.standard_normal((n_paths, t))
    u = rng.random((n_paths, t))
    high = rng.random(n_paths) < pi_high  # S_0 ~ stationary distribution
    sig = np.empty((n_paths, t))
    for j in range(t):
        if j > 0:
            stay = np.where(high, cfg.p_stay_high, cfg.p_stay_low)
            high = np.where(u[:, j] < stay, high, ~high)
        sig[:, j] = np.where(high, sig_high, sig_low)
    return cfg.mu_daily + sig * z


_GENERATORS = {
    "iid_gaussian": _generate_iid_gaussian,
    "iid_student_t": _generate_iid_student_t,
    "ar1": _generate_ar1,
    "garch": _generate_garch,
    "regime_switching": _generate_regime,
}


def generate_paths(cfg: DGPConfig, rng: np.random.Generator, n_paths: int = 1) -> np.ndarray:
    """Generate ``(n_paths, t_steps)`` independent return paths from the DGP."""
    if cfg.family not in _GENERATORS:
        raise ValueError(f"unknown family {cfg.family!r}")
    return _GENERATORS[cfg.family](cfg, rng, n_paths)


def generate(cfg: DGPConfig, rng: np.random.Generator) -> np.ndarray:
    """Generate a single 1-D return path of length ``t_steps``."""
    return generate_paths(cfg, rng, n_paths=1)[0]


# --------------------------------------------------------------------------- #
# config samplers
# --------------------------------------------------------------------------- #
def sample_config(rng: np.random.Generator, family: str, t_steps: int) -> DGPConfig:
    """Draw a fully-recorded random config for ``family``.

    Common to all families: the true annualized Sharpe is sampled uniformly on
    ``SHARPE_ANNUAL_RANGE`` and the unconditional daily vol uniformly on
    ``SIGMA_DAILY_RANGE``; the daily drift is then
    ``mu = SR_ann / sqrt(252) * sigma`` so that the true Sharpe is exact.
    """
    sr_ann = float(rng.uniform(*SHARPE_ANNUAL_RANGE))
    sigma = float(rng.uniform(*SIGMA_DAILY_RANGE))
    mu = sr_ann / math.sqrt(TRADING_DAYS) * sigma
    base = dict(family=family, t_steps=t_steps, mu_daily=mu, sigma_daily=sigma,
                label=family)

    if family == "iid_gaussian":
        return DGPConfig(**base)
    if family == "iid_student_t":
        return DGPConfig(**base, nu=float(rng.choice(NU_CHOICES)))
    if family == "ar1":
        return DGPConfig(**base, phi=float(rng.choice(PHI_CHOICES)))
    if family == "garch":
        alpha = float(rng.uniform(*GARCH_ALPHA_RANGE))
        persistence = float(rng.uniform(*GARCH_PERSISTENCE_RANGE))
        beta = persistence - alpha
        return DGPConfig(**base, garch_alpha=alpha, garch_beta=beta)
    if family == "regime_switching":
        return DGPConfig(
            **base,
            vol_ratio=float(rng.uniform(*VOL_RATIO_RANGE)),
            p_stay_low=float(rng.uniform(*P_STAY_LOW_RANGE)),
            p_stay_high=float(rng.uniform(*P_STAY_HIGH_RANGE)),
        )
    raise ValueError(f"unknown family {family!r}")


def canonical_configs(t_steps: int = 1000) -> dict[str, DGPConfig]:
    """One representative, fixed config per family (mid-range parameters).

    Used by the max-drawdown truth cache, the setup figure, and the tests. The
    true annualized Sharpe is 1.0 and the daily vol 1% for every family, so the
    families differ only in their dependence/tail structure.
    """
    sigma = 0.01
    mu = 1.0 / math.sqrt(TRADING_DAYS) * sigma
    base = dict(t_steps=t_steps, mu_daily=mu, sigma_daily=sigma)
    return {
        "iid_gaussian": DGPConfig(family="iid_gaussian", **base, label="iid_gaussian"),
        "iid_student_t": DGPConfig(family="iid_student_t", **base, nu=6.0,
                                   label="iid_student_t"),
        "ar1": DGPConfig(family="ar1", **base, phi=0.2, label="ar1"),
        "garch": DGPConfig(family="garch", **base, garch_alpha=0.08, garch_beta=0.88,
                           label="garch"),
        "regime_switching": DGPConfig(family="regime_switching", **base, vol_ratio=3.0,
                                      p_stay_low=0.98, p_stay_high=0.95,
                                      label="regime_switching"),
    }
