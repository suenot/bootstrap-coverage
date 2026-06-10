"""Turn experiment records into the paper's quantitative results.

Five questions, all answered with measured numbers:

1. **Headline coverage table.** Empirical coverage of the true Sharpe by
   method x DGP family x T, at 90% and 95% nominal, with Wilson binomial CIs
   on the coverage estimates themselves, plus interval widths.
2. **Honest positive result.** On the truly iid DGPs, is the blog's iid /
   trade-level resampling calibrated? (It should be -- we check.)
3. **Failure modes.** How does coverage degrade with the AR(1) coefficient and
   with GARCH persistence, per method? (The trade-resampling failure mode,
   quantified.)
4. **Block-length sensitivity.** Stationary-bootstrap coverage across the mean
   block length sweep vs the Politis-White automatic choice.
5. **Width at matched coverage.** Among methods not significantly undercovering
   on dependent DGPs, which gives the narrowest intervals?

Plus the max-drawdown study: average realized tail probability
``F_true(q_hat)`` vs nominal q, and the relative bias of the predicted
quantiles. All outputs are JSON-able.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from .methods import MDD_METHODS, MDD_QS, METHOD_ORDER, SB_MEAN_BLOCKS
from .model import DEPENDENT_FAMILIES, FAMILIES, IID_FAMILIES

LEVELS: tuple[float, ...] = (0.90, 0.95)


def to_frame(records: list[dict]) -> pd.DataFrame:
    return pd.DataFrame.from_records(records)


# --------------------------------------------------------------------------- #
# binomial uncertainty on coverage estimates
# --------------------------------------------------------------------------- #
def wilson_ci(k: int, n: int, z: float = 1.959963984540054) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion (95% by default)."""
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    denom = 1.0 + z**2 / n
    center = (p + z**2 / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return (max(0.0, center - half), min(1.0, center + half))


def _cell(sub: pd.DataFrame, method: str, level: float) -> dict:
    tag = f"{method}_{round(level * 100):d}"
    cov = sub[f"cov_{tag}"]
    wid = sub[f"width_{tag}"]
    n = int(cov.notna().sum())
    k = int(cov.sum())
    lo, hi = wilson_ci(k, n)
    return {
        "n": n,
        "covered": k,
        "coverage": k / n if n else float("nan"),
        "wilson_lo": lo,
        "wilson_hi": hi,
        "mean_width": float(wid.mean()),
        "median_width": float(wid.median()),
    }


# --------------------------------------------------------------------------- #
# 1. headline coverage table
# --------------------------------------------------------------------------- #
def coverage_table(df: pd.DataFrame,
                   methods: tuple[str, ...] = METHOD_ORDER,
                   levels: tuple[float, ...] = LEVELS) -> list[dict]:
    """Coverage by method x family x T x level, with Wilson CIs and widths."""
    rows: list[dict] = []
    for family in FAMILIES:
        for t in sorted(df["cfg_t_steps"].unique()):
            sub = df[(df["cfg_family"] == family) & (df["cfg_t_steps"] == t)]
            if sub.empty:
                continue
            for method in methods:
                for level in levels:
                    rows.append({
                        "method": method, "family": family, "t": int(t),
                        "level": level, **_cell(sub, method, level),
                    })
    return rows


def coverage_pivot(df: pd.DataFrame, level: float,
                   methods: tuple[str, ...] = METHOD_ORDER,
                   t: int | None = None) -> pd.DataFrame:
    """Coverage matrix (methods x families) at one level, optionally one T."""
    sub = df if t is None else df[df["cfg_t_steps"] == t]
    data = {}
    for family in FAMILIES:
        fam = sub[sub["cfg_family"] == family]
        data[family] = [
            _cell(fam, m, level)["coverage"] if not fam.empty else float("nan")
            for m in methods
        ]
    return pd.DataFrame(data, index=list(methods))


# --------------------------------------------------------------------------- #
# 2. honest positive: calibration on truly iid DGPs
# --------------------------------------------------------------------------- #
def iid_calibration(df: pd.DataFrame,
                    methods: tuple[str, ...] = METHOD_ORDER,
                    levels: tuple[float, ...] = LEVELS) -> dict:
    """Coverage on iid families pooled over T, per method -- including whether
    the Wilson CI contains the nominal level (calibrated) or sits above/below."""
    sub = df[df["cfg_family"].isin(IID_FAMILIES)]
    out: dict = {"n_experiments": int(len(sub)), "methods": {}}
    for method in methods:
        out["methods"][method] = {}
        for level in levels:
            c = _cell(sub, method, level)
            verdict = "calibrated"
            if c["wilson_hi"] < level:
                verdict = "undercovers"
            elif c["wilson_lo"] > level:
                verdict = "conservative"
            out["methods"][method][f"level_{round(level * 100):d}"] = {**c, "verdict": verdict}
    return out


# --------------------------------------------------------------------------- #
# 3. failure modes: coverage vs dependence strength
# --------------------------------------------------------------------------- #
def coverage_vs_ar_phi(df: pd.DataFrame,
                       methods: tuple[str, ...] = METHOD_ORDER,
                       level: float = 0.90) -> list[dict]:
    """Coverage per AR(1) phi value (pooled over T, plus per-T), per method."""
    sub = df[df["cfg_family"] == "ar1"]
    rows: list[dict] = []
    for phi in sorted(sub["cfg_phi"].dropna().unique()):
        s_phi = sub[sub["cfg_phi"] == phi]
        for method in methods:
            row = {"phi": float(phi), "method": method, "level": level,
                   **_cell(s_phi, method, level)}
            for t in sorted(sub["cfg_t_steps"].unique()):
                st = s_phi[s_phi["cfg_t_steps"] == t]
                row[f"coverage_T{int(t)}"] = _cell(st, method, level)["coverage"]
            rows.append(row)
    return rows


def coverage_vs_garch_persistence(df: pd.DataFrame,
                                  methods: tuple[str, ...] = METHOD_ORDER,
                                  level: float = 0.90,
                                  n_bins: int = 4) -> list[dict]:
    """Coverage per GARCH persistence (alpha+beta) bin, pooled over T."""
    sub = df[df["cfg_family"] == "garch"].copy()
    sub["persistence"] = sub["cfg_garch_alpha"] + sub["cfg_garch_beta"]
    edges = np.quantile(sub["persistence"], np.linspace(0, 1, n_bins + 1))
    sub["pbin"] = pd.cut(sub["persistence"], bins=edges, include_lowest=True)
    rows: list[dict] = []
    for b, grp in sub.groupby("pbin", observed=True):
        if grp.empty:
            continue
        for method in methods:
            rows.append({
                "persistence_bin": str(b),
                "mean_persistence": float(grp["persistence"].mean()),
                "method": method, "level": level,
                **_cell(grp, method, level),
            })
    return rows


# --------------------------------------------------------------------------- #
# 4. block-length sensitivity
# --------------------------------------------------------------------------- #
def block_length_sensitivity(df: pd.DataFrame, level: float = 0.90) -> list[dict]:
    """Stationary-bootstrap coverage/width across the mean-block sweep + auto,
    per family x T; includes the mean automatic block length chosen."""
    methods = [f"sb_l{b}" for b in SB_MEAN_BLOCKS] + ["sb_auto"]
    rows: list[dict] = []
    for family in FAMILIES:
        for t in sorted(df["cfg_t_steps"].unique()):
            sub = df[(df["cfg_family"] == family) & (df["cfg_t_steps"] == t)]
            if sub.empty:
                continue
            for method in methods:
                rows.append({
                    "family": family, "t": int(t), "method": method, "level": level,
                    **_cell(sub, method, level),
                    "mean_auto_block_len": float(sub["x_sb_auto_block_len"].mean()),
                })
    return rows


# --------------------------------------------------------------------------- #
# 5. width at matched coverage
# --------------------------------------------------------------------------- #
def width_at_matched_coverage(df: pd.DataFrame,
                              methods: tuple[str, ...] = METHOD_ORDER,
                              levels: tuple[float, ...] = LEVELS) -> list[dict]:
    """Per T x level over the *dependent* families pooled: which methods are not
    significantly undercovering (Wilson upper bound >= nominal), and among those
    which has the narrowest mean width."""
    dep = df[df["cfg_family"].isin(DEPENDENT_FAMILIES)]
    rows: list[dict] = []
    for t in sorted(dep["cfg_t_steps"].unique()):
        sub = dep[dep["cfg_t_steps"] == t]
        for level in levels:
            cells = {m: _cell(sub, m, level) for m in methods}
            ok = {m: c for m, c in cells.items() if c["wilson_hi"] >= level}
            best = min(ok, key=lambda m: ok[m]["mean_width"]) if ok else None
            for m, c in cells.items():
                rows.append({
                    "t": int(t), "level": level, "method": m, **c,
                    "not_undercovering": int(m in ok),
                    "narrowest_calibrated": int(m == best),
                })
    return rows


# --------------------------------------------------------------------------- #
# max-drawdown study
# --------------------------------------------------------------------------- #
def mdd_summary(mdd_df: pd.DataFrame,
                methods: tuple[str, ...] = MDD_METHODS,
                qs: tuple[float, ...] = MDD_QS) -> list[dict]:
    """Per method x family x T x q: mean/median realized tail probability
    ``F_true(q_hat)`` (nominal value: q) and relative bias of ``q_hat``."""
    rows: list[dict] = []
    for family in sorted(mdd_df["cfg_family"].unique()):
        for t in sorted(mdd_df["cfg_t_steps"].unique()):
            sub = mdd_df[(mdd_df["cfg_family"] == family) & (mdd_df["cfg_t_steps"] == t)]
            if sub.empty:
                continue
            for method in methods:
                for q in qs:
                    tag = f"{method}_q{round(q * 100):d}"
                    probs = sub[f"prob_{tag}"]
                    rels = sub[f"relerr_{tag}"]
                    rows.append({
                        "method": method, "family": family, "t": int(t),
                        "nominal_q": q, "n": int(len(sub)),
                        "mean_realized_prob": float(probs.mean()),
                        "median_realized_prob": float(probs.median()),
                        "mean_relerr_qhat": float(rels.mean()),
                        "mean_abs_relerr_qhat": float(rels.abs().mean()),
                        "true_q": float(sub[f"true_q{round(q * 100):d}"].iloc[0]),
                    })
    return rows


# --------------------------------------------------------------------------- #
# top-level summary
# --------------------------------------------------------------------------- #
def summarize(df: pd.DataFrame, mdd_df: pd.DataFrame | None = None) -> dict:
    out: dict = {
        "n_experiments": int(len(df)),
        "coverage_table": coverage_table(df),
        "iid_calibration": iid_calibration(df),
        "coverage_vs_ar_phi": coverage_vs_ar_phi(df),
        "coverage_vs_garch_persistence": coverage_vs_garch_persistence(df),
        "block_length_sensitivity": block_length_sensitivity(df),
        "width_at_matched_coverage": width_at_matched_coverage(df),
    }
    if mdd_df is not None and not mdd_df.empty:
        out["mdd"] = {
            "n_experiments": int(len(mdd_df)),
            "summary": mdd_summary(mdd_df),
        }
    return out
