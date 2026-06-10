"""Sanity tests for the DGPs, CI methods, and the experiment harness.

Run: python -m pytest -q   (from the project root)

Everything is seeded, so the assertions are deterministic (no flaky tolerances).
"""

from __future__ import annotations

import numpy as np
import pytest

from bootstrap_experiments import (
    DGPConfig,
    canonical_configs,
    generate,
    generate_paths,
    iid_indices,
    jackknife_sharpes,
    lo_iid_ci,
    max_drawdown_paths,
    mdd_bootstrap_quantiles,
    politis_white_block_length,
    regime_sigmas,
    regime_stationary_probs,
    run_batch,
    run_experiment,
    sample_config,
    segment_cusum,
    segment_fixed,
    segment_stats,
    sharpe_annualized,
    sharpe_ci_bundle,
    sharpe_from_indices,
    stationary_bootstrap_indices,
    to_frame,
)


# --------------------------------------------------------------------------- #
# DGP moments and true-Sharpe identities (large-T)
# --------------------------------------------------------------------------- #
def test_iid_gaussian_moments():
    cfg = canonical_configs(t_steps=200_000)["iid_gaussian"]
    r = generate(cfg, np.random.default_rng(0))
    assert abs(r.mean() - cfg.mu_daily) < 4 * cfg.sigma_daily / np.sqrt(r.size)
    assert abs(r.std(ddof=1) - cfg.sigma_daily) / cfg.sigma_daily < 0.01


def test_student_t_unconditional_sd():
    """The standardized Student-t has unit variance -> sd is exactly sigma."""
    cfg = canonical_configs(t_steps=200_000)["iid_student_t"]
    r = generate(cfg, np.random.default_rng(1))
    assert abs(r.std(ddof=1) - cfg.sigma_daily) / cfg.sigma_daily < 0.02
    # heavier tails than Gaussian: excess kurtosis clearly positive (nu=6 -> 3)
    z = (r - r.mean()) / r.std(ddof=1)
    assert np.mean(z**4) - 3.0 > 1.0


def test_ar1_autocorrelation_and_sd():
    cfg = canonical_configs(t_steps=200_000)["ar1"]
    r = generate(cfg, np.random.default_rng(2))
    xc = r - r.mean()
    rho1 = float(xc[:-1] @ xc[1:]) / float(xc @ xc)
    assert abs(rho1 - cfg.phi) < 0.02            # lag-1 autocorr == phi
    assert abs(r.std(ddof=1) - cfg.sigma_daily) / cfg.sigma_daily < 0.02


def test_garch_unconditional_sd_and_clustering():
    cfg = canonical_configs(t_steps=200_000)["garch"]
    r = generate(cfg, np.random.default_rng(3))
    # unconditional variance = omega/(1-alpha-beta) = sigma^2 by construction
    assert abs(r.std(ddof=1) - cfg.sigma_daily) / cfg.sigma_daily < 0.05
    # vol clustering: squared returns positively autocorrelated
    s = r**2 - np.mean(r**2)
    rho1_sq = float(s[:-1] @ s[1:]) / float(s @ s)
    assert rho1_sq > 0.05
    # returns themselves ~uncorrelated
    xc = r - r.mean()
    assert abs(float(xc[:-1] @ xc[1:]) / float(xc @ xc)) < 0.02


def test_regime_switching_sd_matches_mixture_identity():
    cfg = canonical_configs(t_steps=200_000)["regime_switching"]
    pi_low, pi_high = regime_stationary_probs(cfg.p_stay_low, cfg.p_stay_high)
    sig_low, sig_high = regime_sigmas(cfg)
    mix_var = pi_low * sig_low**2 + pi_high * sig_high**2
    assert mix_var == pytest.approx(cfg.sigma_daily**2, rel=1e-12)  # analytic identity
    r = generate(cfg, np.random.default_rng(4))
    assert abs(r.std(ddof=1) - cfg.sigma_daily) / cfg.sigma_daily < 0.05


def test_true_sharpe_recovered_all_families():
    """Sample annualized Sharpe -> configured truth at large T, every family."""
    for name, cfg in canonical_configs(t_steps=300_000).items():
        r = generate(cfg, np.random.default_rng(5))
        assert abs(sharpe_annualized(r) - cfg.true_sharpe_annual) < 0.15, name
        assert cfg.true_sharpe_annual == pytest.approx(1.0)


# --------------------------------------------------------------------------- #
# resampling machinery
# --------------------------------------------------------------------------- #
def test_stationary_bootstrap_basics():
    """Indices valid; geometric blocks have the configured mean length; the
    resampled marginal mean is centered on the sample mean (stationarity of the
    resampled law under the Politis-Romano scheme)."""
    rng = np.random.default_rng(6)
    t, b, mean_block = 2000, 400, 10.0
    idx = stationary_bootstrap_indices(t, b, mean_block, rng)
    assert idx.min() >= 0 and idx.max() < t
    # consecutive-step continuations: P(continue) = 1 - 1/L
    cont = (idx[:, 1:] == (idx[:, :-1] + 1) % t).mean()
    assert abs(cont - (1.0 - 1.0 / mean_block)) < 0.01
    r = np.random.default_rng(7).standard_normal(t)
    resampled_means = r[idx].mean(axis=1)
    assert abs(resampled_means.mean() - r.mean()) < 5e-3


def test_politis_white_iid_vs_ar():
    """Auto block length: small on iid data, larger under positive dependence."""
    rng = np.random.default_rng(8)
    iid = rng.standard_normal(4000)
    cfg = DGPConfig(family="ar1", t_steps=4000, mu_daily=0.0, sigma_daily=0.01, phi=0.3)
    ar = generate(cfg, np.random.default_rng(9))
    bl_iid = politis_white_block_length(iid)
    bl_ar = politis_white_block_length(ar)
    assert bl_iid["b_sb"] < bl_ar["b_sb"]
    assert bl_ar["b_sb"] > 3.0
    assert 1.0 <= bl_iid["b_sb"] <= bl_iid["b_max"]


def test_trade_segmentation():
    rng = np.random.default_rng(10)
    r = rng.standard_normal(103) * 0.01
    seg_sum, seg_ss, seg_len = segment_fixed(r, 5)
    assert seg_sum.size == 20 and (seg_len == 5).all()
    assert seg_sum.sum() == pytest.approx(r[:100].sum())
    bounds = segment_cusum(r, 2.0)
    assert bounds[0] == 0 and bounds[-1] == r.size
    assert (np.diff(bounds) >= 1).all()
    s2, ss2, l2 = segment_stats(r, bounds)
    assert l2.sum() == r.size and s2.sum() == pytest.approx(r.sum())
    thr = 2.0 * r.std(ddof=1)
    for a, b in zip(bounds[:-2], bounds[1:-1]):  # all but the final partial trade
        assert abs(r[a:b].sum()) >= thr - 1e-12


def test_lo_iid_se_matches_bootstrap_se_under_iid():
    cfg = canonical_configs(t_steps=2000)["iid_gaussian"]
    r = generate(cfg, np.random.default_rng(11))
    samples = sharpe_from_indices(r, iid_indices(r.size, 4000, np.random.default_rng(12)))
    boot_se = samples.std(ddof=1)
    ci = lo_iid_ci(r, levels=(0.95,))[0.95]
    lo_se = (ci[1] - ci[0]) / (2 * 1.959963984540054)
    assert abs(lo_se - boot_se) / boot_se < 0.12


def test_jackknife_matches_direct():
    r = np.random.default_rng(13).standard_normal(60) * 0.01 + 0.0005
    jack = jackknife_sharpes(r)
    direct = np.array([sharpe_annualized(np.delete(r, i)) for i in range(r.size)])
    np.testing.assert_allclose(jack, direct, rtol=1e-10)


def test_ci_methods_well_formed():
    cfg = canonical_configs(t_steps=1000)["ar1"]
    r = generate(cfg, np.random.default_rng(14))
    bundle = sharpe_ci_bundle(r, np.random.default_rng(15), n_boot=400)
    for name, m in bundle.items():
        for lvl, (lo, hi) in m["ci"].items():
            assert np.isfinite(lo) and np.isfinite(hi), (name, lvl)
            assert lo < hi, (name, lvl)
        # 95% interval contains the 90% one (same center, nested by construction)
        l90, h90 = m["ci"][0.90]
        l95, h95 = m["ci"][0.95]
        assert l95 <= l90 + 1e-12 and h95 >= h90 - 1e-12, name


# --------------------------------------------------------------------------- #
# calibration sanity: iid DGP -> ~nominal coverage for every method (small run)
# --------------------------------------------------------------------------- #
def test_iid_coverage_near_nominal_small_run():
    recs = run_batch(150, families=("iid_gaussian",), t_grid=(250,),
                     seed=123, n_boot=400, n_jobs=1)
    df = to_frame(recs)
    # methods expected to be ~calibrated on iid data at T=250
    for m in ("lo_iid", "lo_hac", "iid_pct", "iid_basic", "iid_bca",
              "trade_fixed_5", "sb_l5", "sb_auto", "cb_auto"):
        cov95 = df[f"cov_{m}_95"].mean()
        cov90 = df[f"cov_{m}_90"].mean()
        assert cov95 >= 0.87, (m, cov95)   # 150 trials: ~4 binomial sd below 0.95
        assert cov90 >= 0.80, (m, cov90)
        assert df[f"width_{m}_95"].mean() > df[f"width_{m}_90"].mean()


# --------------------------------------------------------------------------- #
# max drawdown
# --------------------------------------------------------------------------- #
def test_max_drawdown_known_path():
    r = np.array([[0.10, -0.50, 1.00, -0.10]])
    # equity: 1.10, 0.55, 1.10, 0.99 -> deepest drawdown (0.55-1.10)/1.10 = -0.5
    assert max_drawdown_paths(r)[0] == pytest.approx(-0.5)
    flat = np.zeros((1, 10))
    assert max_drawdown_paths(flat)[0] == pytest.approx(0.0)


def test_mdd_bootstrap_quantiles_sane():
    cfg = canonical_configs(t_steps=250)["iid_gaussian"]
    r = generate(cfg, np.random.default_rng(16))
    out = mdd_bootstrap_quantiles(r, np.random.default_rng(17), n_boot=300)
    for name, m in out.items():
        q05, q50 = m["q"][0.05], m["q"][0.50]
        assert -1.0 < q05 <= q50 <= 0.0, name


# --------------------------------------------------------------------------- #
# determinism
# --------------------------------------------------------------------------- #
def _records_identical(a: dict, b: dict) -> bool:
    """Bit-identical comparison treating NaN == NaN (unused-param fields)."""
    if a.keys() != b.keys():
        return False
    for k in a:
        x, y = a[k], b[k]
        if isinstance(x, float) and isinstance(y, float) and np.isnan(x) and np.isnan(y):
            continue
        if x != y:
            return False
    return True


def test_run_experiment_deterministic():
    rng_a = np.random.default_rng(np.random.SeedSequence(99))
    cfg_a = sample_config(rng_a, "garch", 250)
    rec_a = run_experiment(cfg_a, rng_a, n_boot=200)
    rng_b = np.random.default_rng(np.random.SeedSequence(99))
    cfg_b = sample_config(rng_b, "garch", 250)
    rec_b = run_experiment(cfg_b, rng_b, n_boot=200)
    assert _records_identical(cfg_a.as_record(), cfg_b.as_record())
    assert _records_identical(rec_a, rec_b)  # bit-identical, every field


def test_run_batch_deterministic_and_complete():
    recs1 = run_batch(2, families=("ar1", "garch"), t_grid=(250,), seed=7,
                      n_boot=150, n_jobs=1)
    recs2 = run_batch(2, families=("ar1", "garch"), t_grid=(250,), seed=7,
                      n_boot=150, n_jobs=1)
    assert all(_records_identical(r1, r2) for r1, r2 in zip(recs1, recs2))
    df = to_frame(recs1)
    assert len(df) == 4
    for col in ("cov_iid_pct_95", "cov_sb_auto_90", "width_trade_cusum_95",
                "true_sharpe", "sharpe_hat", "x_sb_auto_block_len"):
        assert col in df.columns
