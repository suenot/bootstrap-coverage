"""Confidence-interval constructors for the annualized Sharpe ratio (and
bootstrap quantile predictors for the maximum-drawdown distribution).

Methods evaluated (all target the same parameter, the population annualized
Sharpe of the daily return stream):

1. **iid bootstrap of returns** -- resample the T daily returns with
   replacement; percentile, basic, and BCa intervals from the same bootstrap
   distribution.
2. **Stationary bootstrap** (Politis & Romano 1994) -- geometric block lengths
   with mean L, wrapping circularly; we sweep L in ``SB_MEAN_BLOCKS`` and also
   use the Politis-White (2004, with the Patton-Politis-White 2009 correction)
   automatic block length. Percentile intervals.
3. **Circular block bootstrap** (Politis & Romano 1992) -- fixed-length blocks
   with the Politis-White automatic length. Percentile intervals.
4. **Trade-level iid resampling** -- the blog post's actual recipe. The daily
   stream is segmented into "trades" and the trades are resampled iid with
   replacement. Two segmentations, both documented exactly:

   * *fixed-length episodes*: consecutive non-overlapping windows of
     ``TRADE_FIXED_LENGTHS`` days (a remainder of < L days at the end is
     dropped and its size recorded). This is the blog's method under the most
     charitable consistent reading: a trade = a position episode of roughly
     constant duration.
   * *threshold-crossing episodes* (``trade_cusum``): a trade starts at the
     first day after the previous trade closed and closes at the end of the
     first day on which the absolute cumulative return since the trade opened
     reaches ``CUSUM_MULT`` x (sample sd of daily returns) -- a take-profit /
     stop-loss style exit. The final partial episode is kept as a trade.

   In both cases we resample whole trades with replacement, concatenate the
   underlying daily returns, and recompute the *daily-frequency* annualized
   Sharpe, so the resampled statistic targets the same parameter as every other
   method. (The blog literally computes ``mean/std*sqrt(252)`` on *per-trade*
   returns, which targets a different parameter whenever trades span more than
   one day; scoring that version would conflate a units error with the
   resampling question, so we score the charitable version. Fixed-length trade
   resampling is exactly a non-overlapping-block iid bootstrap.)
5. **Analytic** -- Lo (2002): the iid standard error
   ``SE = sqrt((1 + SR_d^2/2)/T)`` (normal returns), and its
   autocorrelation-adjusted variant via the delta method with a Newey-West/HAC
   long-run covariance of ``(r_t, r_t^2)`` (Lo's GMM estimator), lag
   ``floor(4*(T/100)^(2/9))``. Normal-theory intervals.

All bootstrap methods use the same number of replications ``n_boot`` and a
caller-supplied seeded Generator; everything is deterministic.
"""

from __future__ import annotations

import math

import numpy as np
from scipy import stats

from .model import TRADING_DAYS

SB_MEAN_BLOCKS: tuple[int, ...] = (5, 10, 20, 50)   # stationary-bootstrap mean L sweep
TRADE_FIXED_LENGTHS: tuple[int, ...] = (5, 21)      # fixed "trade" episode lengths
CUSUM_MULT: float = 2.0                              # threshold-crossing exit, in sd units
DEFAULT_LEVELS: tuple[float, ...] = (0.90, 0.95)
DEFAULT_N_BOOT: int = 2000

#: canonical column order for tables and figures
METHOD_ORDER: tuple[str, ...] = (
    "lo_iid", "lo_hac",
    "iid_pct", "iid_basic", "iid_bca",
    "trade_fixed_5", "trade_fixed_21", "trade_cusum",
    "sb_l5", "sb_l10", "sb_l20", "sb_l50", "sb_auto",
    "cb_auto",
)


def method_parameters() -> dict:
    """The fixed method-level constants (recorded in results meta)."""
    return {
        "sb_mean_blocks": SB_MEAN_BLOCKS,
        "trade_fixed_lengths": TRADE_FIXED_LENGTHS,
        "cusum_mult": CUSUM_MULT,
        "default_n_boot": DEFAULT_N_BOOT,
        "levels": DEFAULT_LEVELS,
    }


# --------------------------------------------------------------------------- #
# the statistic
# --------------------------------------------------------------------------- #
def sharpe_annualized(r: np.ndarray) -> float:
    """Annualized Sharpe ``sqrt(252) * mean / sd`` (sd with ddof=1)."""
    return float(math.sqrt(TRADING_DAYS) * r.mean() / r.std(ddof=1))


def _sharpe_from_sums(s: np.ndarray, ss: np.ndarray, n: np.ndarray) -> np.ndarray:
    """Vectorized annualized Sharpe from per-resample (sum, sum-of-squares, n)."""
    mean = s / n
    var = (ss - s * s / n) / (n - 1.0)
    return math.sqrt(TRADING_DAYS) * mean / np.sqrt(var)


def sharpe_from_indices(r: np.ndarray, idx: np.ndarray) -> np.ndarray:
    """Annualized Sharpe of each resampled row ``r[idx[b]]`` (idx: (B, T))."""
    vals = r[idx]
    m = vals.mean(axis=1)
    sd = vals.std(axis=1, ddof=1)
    return math.sqrt(TRADING_DAYS) * m / sd


# --------------------------------------------------------------------------- #
# interval constructors from a bootstrap distribution
# --------------------------------------------------------------------------- #
def ci_percentile(samples: np.ndarray, levels: tuple[float, ...]) -> dict[float, tuple[float, float]]:
    out: dict[float, tuple[float, float]] = {}
    for lvl in levels:
        a = (1.0 - lvl) / 2.0
        lo, hi = np.quantile(samples, [a, 1.0 - a])
        out[lvl] = (float(lo), float(hi))
    return out


def ci_basic(samples: np.ndarray, theta_hat: float,
             levels: tuple[float, ...]) -> dict[float, tuple[float, float]]:
    out: dict[float, tuple[float, float]] = {}
    for lvl in levels:
        a = (1.0 - lvl) / 2.0
        q_lo, q_hi = np.quantile(samples, [a, 1.0 - a])
        out[lvl] = (float(2.0 * theta_hat - q_hi), float(2.0 * theta_hat - q_lo))
    return out


def jackknife_sharpes(r: np.ndarray) -> np.ndarray:
    """Leave-one-out annualized Sharpe values, computed in O(T) via sums."""
    t = r.size
    s, ss = r.sum(), float(r @ r)
    mean_i = (s - r) / (t - 1)
    var_i = (ss - r * r - (t - 1) * mean_i**2) / (t - 2)
    return math.sqrt(TRADING_DAYS) * mean_i / np.sqrt(var_i)


def ci_bca(samples: np.ndarray, theta_hat: float, jack: np.ndarray,
           levels: tuple[float, ...]) -> dict[float, tuple[float, float]]:
    """BCa interval (Efron 1987): bias correction z0 + jackknife acceleration."""
    b = samples.size
    frac = float(np.clip(np.mean(samples < theta_hat), 1.0 / (b + 1), 1.0 - 1.0 / (b + 1)))
    z0 = float(stats.norm.ppf(frac))
    d = jack.mean() - jack
    denom = float(np.sum(d**2)) ** 1.5
    accel = float(np.sum(d**3)) / (6.0 * denom) if denom > 0 else 0.0
    out: dict[float, tuple[float, float]] = {}
    for lvl in levels:
        a = (1.0 - lvl) / 2.0
        bounds = []
        for z_a in (stats.norm.ppf(a), stats.norm.ppf(1.0 - a)):
            adj = stats.norm.cdf(z0 + (z0 + z_a) / (1.0 - accel * (z0 + z_a)))
            bounds.append(float(np.quantile(samples, float(np.clip(adj, 0.0, 1.0)))))
        out[lvl] = (min(bounds), max(bounds))
    return out


# --------------------------------------------------------------------------- #
# resampling index generators
# --------------------------------------------------------------------------- #
def iid_indices(t_len: int, n_boot: int, rng: np.random.Generator) -> np.ndarray:
    return rng.integers(0, t_len, size=(n_boot, t_len))


def stationary_bootstrap_indices(t_len: int, n_boot: int, mean_block: float,
                                 rng: np.random.Generator) -> np.ndarray:
    """Politis-Romano stationary bootstrap indices (geometric blocks, circular wrap)."""
    p = min(1.0, 1.0 / max(mean_block, 1.0))
    starts = rng.integers(0, t_len, size=(n_boot, t_len))
    restart = rng.random((n_boot, t_len)) < p
    idx = np.empty((n_boot, t_len), dtype=np.int64)
    prev = starts[:, 0].copy()
    idx[:, 0] = prev
    for j in range(1, t_len):
        nxt = prev + 1
        nxt[nxt == t_len] = 0
        prev = np.where(restart[:, j], starts[:, j], nxt)
        idx[:, j] = prev
    return idx


def circular_block_indices(t_len: int, n_boot: int, block_len: int,
                           rng: np.random.Generator) -> np.ndarray:
    """Circular block bootstrap indices: fixed blocks of ``block_len``, wrapped."""
    b = max(1, int(round(block_len)))
    n_blocks = int(math.ceil(t_len / b))
    starts = rng.integers(0, t_len, size=(n_boot, n_blocks))
    offsets = np.arange(b)
    idx = (starts[:, :, None] + offsets[None, None, :]) % t_len
    return idx.reshape(n_boot, n_blocks * b)[:, :t_len]


# --------------------------------------------------------------------------- #
# Politis-White (2004) automatic block length, with the 2009 Patton correction
# --------------------------------------------------------------------------- #
def politis_white_block_length(x: np.ndarray) -> dict[str, float]:
    """Automatic optimal block lengths for the stationary (``b_sb``) and circular
    (``b_cb``) block bootstraps, following Politis & White (2004) as corrected by
    Patton, Politis & White (2009)."""
    n = x.size
    kn = max(5, int(math.ceil(math.log10(n))))
    mmax = min(int(math.ceil(math.sqrt(n))) + kn, n - 1)
    b_max = float(math.ceil(min(3.0 * math.sqrt(n), n / 3.0)))

    xc = x - x.mean()
    acov = np.array([float(xc[: n - k] @ xc[k:]) for k in range(mmax + 1)]) / n
    if acov[0] <= 0:
        return {"b_sb": 1.0, "b_cb": 1.0, "m_hat": 0.0, "b_max": b_max}
    rho = acov / acov[0]

    band = 2.0 * math.sqrt(math.log10(n) / n)
    insig = np.abs(rho[1:]) < band  # insig[k-1] <-> lag k
    m_hat: int | None = None
    for m in range(1, len(insig) - kn + 2):
        if insig[m - 1: m - 1 + kn].all():  # lags m..m+kn-1 all insignificant
            m_hat = m - 1
            break
    if m_hat is None:
        sig_lags = np.flatnonzero(~insig) + 1
        m_hat = int(sig_lags.max()) if sig_lags.size else 0

    big_m = min(2 * m_hat, mmax)
    if big_m == 0:
        return {"b_sb": 1.0, "b_cb": 1.0, "m_hat": float(m_hat), "b_max": b_max}

    k = np.arange(1, big_m + 1)
    frac = k / big_m
    lam = np.where(frac <= 0.5, 1.0, 2.0 * (1.0 - frac))  # trapezoidal taper
    g_hat = 2.0 * float(np.sum(lam * k * acov[1: big_m + 1]))
    g0 = float(acov[0] + 2.0 * np.sum(lam * acov[1: big_m + 1]))  # long-run var
    if g0 <= 0 or g_hat == 0.0:
        return {"b_sb": 1.0, "b_cb": 1.0, "m_hat": float(m_hat), "b_max": b_max}
    d_sb = 2.0 * g0**2
    d_cb = (4.0 / 3.0) * g0**2
    b_sb = ((2.0 * g_hat**2) / d_sb) ** (1.0 / 3.0) * n ** (1.0 / 3.0)
    b_cb = ((2.0 * g_hat**2) / d_cb) ** (1.0 / 3.0) * n ** (1.0 / 3.0)
    return {
        "b_sb": float(np.clip(b_sb, 1.0, b_max)),
        "b_cb": float(np.clip(b_cb, 1.0, b_max)),
        "m_hat": float(m_hat),
        "b_max": b_max,
    }


# --------------------------------------------------------------------------- #
# trade segmentation (the blog's unit of resampling)
# --------------------------------------------------------------------------- #
def segment_fixed(r: np.ndarray, length: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fixed-length episodes: returns per-trade (sum, sum_sq, len); remainder dropped."""
    k = r.size // length
    if k < 2:
        raise ValueError(f"too few trades: T={r.size}, trade length={length}")
    trimmed = r[: k * length].reshape(k, length)
    return trimmed.sum(axis=1), (trimmed**2).sum(axis=1), np.full(k, length, dtype=np.int64)


def segment_cusum(r: np.ndarray, mult: float = CUSUM_MULT) -> np.ndarray:
    """Threshold-crossing episode boundaries (start indices, then ``r.size``).

    A trade closes at the end of the first day on which |cumulative return since
    the trade opened| >= ``mult`` * sd(r). The final partial episode is kept.
    """
    thr = mult * float(r.std(ddof=1))
    bounds = [0]
    c = 0.0
    for i in range(r.size):
        c += float(r[i])
        if abs(c) >= thr:
            bounds.append(i + 1)
            c = 0.0
    if bounds[-1] != r.size:
        bounds.append(r.size)
    return np.asarray(bounds, dtype=np.int64)


def segment_stats(r: np.ndarray, bounds: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-segment (sum, sum_sq, len) for boundary array from :func:`segment_cusum`."""
    starts = bounds[:-1]
    seg_sum = np.add.reduceat(r, starts)
    seg_ss = np.add.reduceat(r * r, starts)
    seg_len = np.diff(bounds)
    return seg_sum, seg_ss, seg_len


def trade_resample_sharpes(seg_sum: np.ndarray, seg_ss: np.ndarray, seg_len: np.ndarray,
                           rng: np.random.Generator, n_boot: int) -> np.ndarray:
    """Resample whole trades iid with replacement; Sharpe of the concatenated stream."""
    k = seg_sum.size
    idx = rng.integers(0, k, size=(n_boot, k))
    s = seg_sum[idx].sum(axis=1)
    ss = seg_ss[idx].sum(axis=1)
    n = seg_len[idx].sum(axis=1).astype(np.float64)
    return _sharpe_from_sums(s, ss, n)


# --------------------------------------------------------------------------- #
# analytic (Lo 2002) intervals
# --------------------------------------------------------------------------- #
def _normal_ci(theta: float, se: float, levels: tuple[float, ...]) -> dict[float, tuple[float, float]]:
    out: dict[float, tuple[float, float]] = {}
    for lvl in levels:
        z = float(stats.norm.ppf(0.5 + lvl / 2.0))
        out[lvl] = (theta - z * se, theta + z * se)
    return out


def lo_iid_ci(r: np.ndarray, levels: tuple[float, ...] = DEFAULT_LEVELS) -> dict[float, tuple[float, float]]:
    """Lo (2002) iid-normal SE: ``SE(SR_d) = sqrt((1 + SR_d^2/2)/T)``, annualized."""
    t = r.size
    theta = sharpe_annualized(r)
    sr_d = theta / math.sqrt(TRADING_DAYS)
    se = math.sqrt(TRADING_DAYS) * math.sqrt((1.0 + 0.5 * sr_d**2) / t)
    return _normal_ci(theta, se, levels)


def newey_west_lags(t: int) -> int:
    return int(math.floor(4.0 * (t / 100.0) ** (2.0 / 9.0)))


def lo_hac_ci(r: np.ndarray, levels: tuple[float, ...] = DEFAULT_LEVELS,
              n_lags: int | None = None) -> dict[float, tuple[float, float]]:
    """Lo (2002) autocorrelation-robust SE: delta method on ``(mean, E[r^2])`` with
    a Newey-West (Bartlett) HAC long-run covariance."""
    t = r.size
    if n_lags is None:
        n_lags = newey_west_lags(t)
    mu = float(r.mean())
    m2 = float(np.mean(r * r))
    var = m2 - mu**2
    sig = math.sqrt(var)
    h = np.column_stack([r - mu, r * r - m2])
    omega = h.T @ h / t
    for k in range(1, n_lags + 1):
        w = 1.0 - k / (n_lags + 1.0)
        gam = h[k:].T @ h[:-k] / t
        omega = omega + w * (gam + gam.T)
    grad = np.array([1.0 / sig + mu**2 / sig**3, -mu / (2.0 * sig**3)])
    var_sr_d = float(grad @ omega @ grad) / t
    se = math.sqrt(TRADING_DAYS * max(var_sr_d, 0.0))
    theta = sharpe_annualized(r)
    return _normal_ci(theta, se, levels)


# --------------------------------------------------------------------------- #
# the full bundle: every method on one sample
# --------------------------------------------------------------------------- #
def sharpe_ci_bundle(r: np.ndarray, rng: np.random.Generator,
                     n_boot: int = DEFAULT_N_BOOT,
                     levels: tuple[float, ...] = DEFAULT_LEVELS) -> dict[str, dict]:
    """Build every method's CIs for the annualized Sharpe of ``r``.

    Returns ``{method_name: {"ci": {level: (lo, hi)}, "extra": {...}}}``. The
    iid percentile/basic/BCa intervals share one bootstrap distribution. RNG
    consumption order is fixed, so results are deterministic given the seed.
    """
    t = r.size
    theta = sharpe_annualized(r)
    out: dict[str, dict] = {}

    out["lo_iid"] = {"ci": lo_iid_ci(r, levels), "extra": {}}
    out["lo_hac"] = {"ci": lo_hac_ci(r, levels), "extra": {"nw_lags": newey_west_lags(t)}}

    # --- iid bootstrap of returns (one distribution, three constructions) ----
    samp = sharpe_from_indices(r, iid_indices(t, n_boot, rng))
    jack = jackknife_sharpes(r)
    out["iid_pct"] = {"ci": ci_percentile(samp, levels), "extra": {}}
    out["iid_basic"] = {"ci": ci_basic(samp, theta, levels), "extra": {}}
    out["iid_bca"] = {"ci": ci_bca(samp, theta, jack, levels), "extra": {}}

    # --- trade-level iid resampling (the blog's method) ----------------------
    for length in TRADE_FIXED_LENGTHS:
        seg_sum, seg_ss, seg_len = segment_fixed(r, length)
        s = trade_resample_sharpes(seg_sum, seg_ss, seg_len, rng, n_boot)
        out[f"trade_fixed_{length}"] = {
            "ci": ci_percentile(s, levels),
            "extra": {"n_trades": int(seg_sum.size), "n_dropped_days": int(t % length)},
        }
    bounds = segment_cusum(r, CUSUM_MULT)
    seg_sum, seg_ss, seg_len = segment_stats(r, bounds)
    s = trade_resample_sharpes(seg_sum, seg_ss, seg_len, rng, n_boot)
    out["trade_cusum"] = {
        "ci": ci_percentile(s, levels),
        "extra": {"n_trades": int(seg_sum.size),
                  "mean_trade_len": float(seg_len.mean())},
    }

    # --- block bootstraps -----------------------------------------------------
    bl = politis_white_block_length(r)
    for mean_block in SB_MEAN_BLOCKS:
        idx = stationary_bootstrap_indices(t, n_boot, float(mean_block), rng)
        out[f"sb_l{mean_block}"] = {
            "ci": ci_percentile(sharpe_from_indices(r, idx), levels),
            "extra": {},
        }
    idx = stationary_bootstrap_indices(t, n_boot, bl["b_sb"], rng)
    out["sb_auto"] = {
        "ci": ci_percentile(sharpe_from_indices(r, idx), levels),
        "extra": {"block_len": bl["b_sb"]},
    }
    idx = circular_block_indices(t, n_boot, int(round(bl["b_cb"])), rng)
    out["cb_auto"] = {
        "ci": ci_percentile(sharpe_from_indices(r, idx), levels),
        "extra": {"block_len": bl["b_cb"]},
    }
    return out


# --------------------------------------------------------------------------- #
# maximum drawdown: statistic + bootstrap quantile predictions
# --------------------------------------------------------------------------- #
MDD_METHODS: tuple[str, ...] = ("iid_pct", "trade_fixed_5", "sb_l20", "sb_auto", "cb_auto")
MDD_QS: tuple[float, ...] = (0.05, 0.50)


def max_drawdown_paths(returns: np.ndarray) -> np.ndarray:
    """Max drawdown (a negative number) of each row, on compounded equity curves."""
    eq = np.cumprod(1.0 + returns, axis=1)
    peak = np.maximum.accumulate(eq, axis=1)
    return ((eq - peak) / peak).min(axis=1)


def mdd_bootstrap_quantiles(r: np.ndarray, rng: np.random.Generator,
                            n_boot: int = DEFAULT_N_BOOT,
                            qs: tuple[float, ...] = MDD_QS) -> dict[str, dict]:
    """Each method's bootstrap prediction of the q-quantiles of the max-drawdown
    *sampling distribution* (over fresh length-T realizations).

    Only methods whose resampled path has length exactly T are included, since
    the max-drawdown distribution depends on path length (this is why
    ``trade_cusum`` and ``trade_fixed_21`` are excluded -- their resampled
    streams have variable / truncated length). Requires T divisible by 5.
    """
    t = r.size
    if t % 5 != 0:
        raise ValueError("mdd bundle requires T divisible by 5 (trade_fixed_5 paths)")
    bl = politis_white_block_length(r)
    out: dict[str, dict] = {}

    def record(name: str, idx: np.ndarray, extra: dict) -> None:
        mdds = max_drawdown_paths(r[idx])
        out[name] = {"q": {q: float(np.quantile(mdds, q)) for q in qs}, "extra": extra}

    record("iid_pct", iid_indices(t, n_boot, rng), {})
    # trade_fixed_5: resample K = T/5 trades, concatenate -> path length exactly T
    k, length = t // 5, 5
    block_starts = np.arange(k) * length
    pick = rng.integers(0, k, size=(n_boot, k))
    idx = (block_starts[pick][:, :, None] + np.arange(length)[None, None, :]).reshape(n_boot, t)
    record("trade_fixed_5", idx, {"n_trades": k})
    record("sb_l20", stationary_bootstrap_indices(t, n_boot, 20.0, rng), {})
    record("sb_auto", stationary_bootstrap_indices(t, n_boot, bl["b_sb"], rng),
           {"block_len": bl["b_sb"]})
    record("cb_auto", circular_block_indices(t, n_boot, int(round(bl["b_cb"])), rng),
           {"block_len": bl["b_cb"]})
    return out
