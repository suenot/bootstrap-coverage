"""Do bootstrap confidence intervals for backtest statistics actually cover?

A controlled coverage study of iid / trade-level / block bootstrap and analytic
(Lo 2002) confidence intervals for the annualized Sharpe ratio (and bootstrap
quantile predictions for the maximum drawdown) under data-generating processes
with serial dependence and analytically known true Sharpe.
"""

from .analysis import (
    block_length_sensitivity,
    coverage_pivot,
    coverage_table,
    coverage_vs_ar_phi,
    coverage_vs_garch_persistence,
    iid_calibration,
    mdd_summary,
    summarize,
    to_frame,
    width_at_matched_coverage,
    wilson_ci,
)
from .methods import (
    CUSUM_MULT,
    DEFAULT_LEVELS,
    DEFAULT_N_BOOT,
    MDD_METHODS,
    MDD_QS,
    METHOD_ORDER,
    SB_MEAN_BLOCKS,
    TRADE_FIXED_LENGTHS,
    ci_basic,
    ci_bca,
    ci_percentile,
    circular_block_indices,
    iid_indices,
    jackknife_sharpes,
    lo_hac_ci,
    lo_iid_ci,
    max_drawdown_paths,
    mdd_bootstrap_quantiles,
    method_parameters,
    politis_white_block_length,
    segment_cusum,
    segment_fixed,
    segment_stats,
    sharpe_annualized,
    sharpe_ci_bundle,
    sharpe_from_indices,
    stationary_bootstrap_indices,
    trade_resample_sharpes,
)
from .model import (
    DEPENDENT_FAMILIES,
    FAMILIES,
    GARCH_BURN_IN,
    IID_FAMILIES,
    T_GRID,
    TRADING_DAYS,
    DGPConfig,
    canonical_configs,
    generate,
    generate_paths,
    regime_sigmas,
    regime_stationary_probs,
    sample_config,
    sampling_ranges,
)
from .simulate import (
    MDD_T_GRID,
    ensure_mdd_truth_cache,
    mdd_truth_distribution,
    run_batch,
    run_experiment,
    run_mdd_batch,
    run_mdd_experiment,
)

__all__ = [
    # model
    "DGPConfig", "FAMILIES", "IID_FAMILIES", "DEPENDENT_FAMILIES", "T_GRID",
    "TRADING_DAYS", "GARCH_BURN_IN", "canonical_configs", "generate",
    "generate_paths", "sample_config", "sampling_ranges",
    "regime_stationary_probs", "regime_sigmas",
    # methods
    "METHOD_ORDER", "MDD_METHODS", "MDD_QS", "SB_MEAN_BLOCKS",
    "TRADE_FIXED_LENGTHS", "CUSUM_MULT", "DEFAULT_LEVELS", "DEFAULT_N_BOOT",
    "sharpe_annualized", "sharpe_from_indices", "sharpe_ci_bundle",
    "ci_percentile", "ci_basic", "ci_bca", "jackknife_sharpes",
    "iid_indices", "stationary_bootstrap_indices", "circular_block_indices",
    "politis_white_block_length", "segment_fixed", "segment_cusum",
    "segment_stats", "trade_resample_sharpes", "lo_iid_ci", "lo_hac_ci",
    "max_drawdown_paths", "mdd_bootstrap_quantiles", "method_parameters",
    # simulate
    "run_experiment", "run_batch", "run_mdd_experiment", "run_mdd_batch",
    "mdd_truth_distribution", "ensure_mdd_truth_cache", "MDD_T_GRID",
    # analysis
    "to_frame", "wilson_ci", "coverage_table", "coverage_pivot",
    "iid_calibration", "coverage_vs_ar_phi", "coverage_vs_garch_persistence",
    "block_length_sensitivity", "width_at_matched_coverage", "mdd_summary",
    "summarize",
]
__version__ = "0.1.0"
