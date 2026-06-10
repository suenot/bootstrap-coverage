"""Generate the paper's figures (vector PDF) from the saved results.

    python -m bootstrap_experiments.figures      # writes paper/figures/*.pdf

Reads results/records.csv and results/results.json; the setup figure
regenerates small illustrative sample paths deterministically.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import TwoSlopeNorm

from .analysis import coverage_pivot, wilson_ci
from .methods import METHOD_ORDER, SB_MEAN_BLOCKS
from .model import DEPENDENT_FAMILIES, FAMILIES, canonical_configs, generate

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
FIGDIR = ROOT / "paper" / "figures"

plt.rcParams.update({
    "font.family": "serif", "font.size": 9, "axes.titlesize": 9,
    "axes.labelsize": 9, "figure.dpi": 120, "savefig.bbox": "tight",
    "axes.spines.top": False, "axes.spines.right": False,
})

FAMILY_LABELS = {
    "iid_gaussian": "iid Gaussian", "iid_student_t": "iid Student-t",
    "ar1": "AR(1)", "garch": "GARCH(1,1)", "regime_switching": "regime vol",
}
FAMILY_COLORS = {
    "iid_gaussian": "#9aa6b2", "iid_student_t": "#e0a458",
    "ar1": "#c0392b", "garch": "#1f3b73", "regime_switching": "#2e8b57",
}
METHOD_LABELS = {
    "lo_iid": "Lo iid SE", "lo_hac": "Lo HAC SE",
    "iid_pct": "iid boot (pct)", "iid_basic": "iid boot (basic)",
    "iid_bca": "iid boot (BCa)",
    "trade_fixed_5": "trade resample (5d)", "trade_fixed_21": "trade resample (21d)",
    "trade_cusum": "trade resample (cusum)",
    "sb_l5": "stationary L=5", "sb_l10": "stationary L=10",
    "sb_l20": "stationary L=20", "sb_l50": "stationary L=50",
    "sb_auto": "stationary auto", "cb_auto": "circular auto",
}
DEGRADATION_METHODS = ("iid_pct", "trade_fixed_5", "lo_iid", "lo_hac", "sb_auto", "cb_auto")
DEGRADATION_COLORS = {
    "iid_pct": "#c0392b", "trade_fixed_5": "#e0a458", "lo_iid": "#9aa6b2",
    "lo_hac": "#7c4d8f", "sb_auto": "#1f3b73", "cb_auto": "#2e8b57",
}


def _sample_autocorr(x: np.ndarray, max_lag: int) -> np.ndarray:
    xc = x - x.mean()
    denom = float(xc @ xc)
    return np.array([float(xc[: x.size - k] @ xc[k:]) / denom
                     for k in range(1, max_lag + 1)])


# --------------------------------------------------------------------------- #
# Fig 1: setup -- DGP sample paths + their dependence structure
# --------------------------------------------------------------------------- #
def fig_setup(path: Path) -> None:
    t = 1000
    cfgs = canonical_configs(t_steps=t)
    rng = np.random.default_rng(np.random.SeedSequence(42))
    paths = {name: generate(cfg, rng) for name, cfg in cfgs.items()}

    fig, axes = plt.subplots(1, 3, figsize=(11, 3.0))

    ax = axes[0]
    for i, (name, r) in enumerate(paths.items()):
        eq = np.cumsum(np.log1p(r)) + 0.12 * i
        ax.plot(eq, lw=0.9, color=FAMILY_COLORS[name], label=FAMILY_LABELS[name])
    ax.set_title("(a) cumulative log-equity (offset), true SR$=1$")
    ax.set_xlabel("day"); ax.set_ylabel("log equity (offset)")
    ax.legend(fontsize=6.5, loc="upper left")

    max_lag = 25
    ax = axes[1]
    for name in ("iid_gaussian", "ar1", "garch", "regime_switching"):
        ax.plot(np.arange(1, max_lag + 1), _sample_autocorr(paths[name], max_lag),
                "o-", ms=2.5, lw=0.9, color=FAMILY_COLORS[name],
                label=FAMILY_LABELS[name])
    band = 1.96 / np.sqrt(t)
    ax.axhspan(-band, band, color="0.85", alpha=0.5, zorder=0)
    ax.axhline(0, color="k", lw=0.6)
    ax.set_title("(b) ACF of returns")
    ax.set_xlabel("lag"); ax.set_ylabel("autocorrelation")
    ax.legend(fontsize=6.5)

    ax = axes[2]
    for name in ("iid_gaussian", "ar1", "garch", "regime_switching"):
        ax.plot(np.arange(1, max_lag + 1), _sample_autocorr(paths[name] ** 2, max_lag),
                "o-", ms=2.5, lw=0.9, color=FAMILY_COLORS[name],
                label=FAMILY_LABELS[name])
    ax.axhspan(-band, band, color="0.85", alpha=0.5, zorder=0)
    ax.axhline(0, color="k", lw=0.6)
    ax.set_title("(c) ACF of squared returns (vol clustering)")
    ax.set_xlabel("lag"); ax.set_ylabel("autocorrelation")
    fig.tight_layout(); fig.savefig(path); plt.close(fig)


# --------------------------------------------------------------------------- #
# Fig 2: coverage heatmap, method x (family, T), 90% nominal
# --------------------------------------------------------------------------- #
def fig_coverage_heatmap(path: Path, df: pd.DataFrame, level: float = 0.90) -> None:
    t_values = sorted(df["cfg_t_steps"].unique())
    mats, col_labels = [], []
    for family in FAMILIES:
        for t in t_values:
            piv = coverage_pivot(df[df["cfg_family"] == family], level, t=int(t))
            mats.append(piv[family].to_numpy())
            col_labels.append(f"{FAMILY_LABELS[family]}\nT={int(t)}")
    mat = np.column_stack(mats)

    fig, ax = plt.subplots(figsize=(11.5, 5.0))
    norm = TwoSlopeNorm(vmin=max(0.4, np.nanmin(mat) - 0.02), vcenter=level, vmax=1.0)
    im = ax.imshow(mat, cmap="RdBu", norm=norm, aspect="auto")
    ax.set_xticks(range(len(col_labels)), col_labels, fontsize=6.5)
    ax.set_yticks(range(len(METHOD_ORDER)),
                  [METHOD_LABELS[m] for m in METHOD_ORDER], fontsize=7.5)
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            v = mat[i, j]
            ax.text(j, i, f"{v:.2f}"[1:] if v < 1 else "1.0", ha="center", va="center",
                    fontsize=6, color="white" if abs(v - level) > 0.12 else "black")
    for j in range(len(t_values), mat.shape[1], len(t_values)):
        ax.axvline(j - 0.5, color="k", lw=1.0)
    cb = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.01)
    cb.set_label(f"empirical coverage (nominal {level:.2f})", fontsize=8)
    ax.set_title(f"Empirical coverage of the true Sharpe, {level:.0%} nominal "
                 "(blue = over, red = under)")
    fig.tight_layout(); fig.savefig(path); plt.close(fig)


# --------------------------------------------------------------------------- #
# Fig 3: degradation -- coverage vs AR phi and vs GARCH persistence
# --------------------------------------------------------------------------- #
def fig_degradation(path: Path, results: dict, level: float = 0.90) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(9.4, 3.4))

    ax = axes[0]
    rows = pd.DataFrame(results["coverage_vs_ar_phi"])
    for m in DEGRADATION_METHODS:
        sub = rows[rows["method"] == m].sort_values("phi")
        err = np.vstack([sub["coverage"] - sub["wilson_lo"],
                         sub["wilson_hi"] - sub["coverage"]])
        ax.errorbar(sub["phi"], sub["coverage"], yerr=err, fmt="o-", ms=3, lw=1.1,
                    capsize=2, color=DEGRADATION_COLORS[m], label=METHOD_LABELS[m])
    ax.axhline(level, color="k", lw=0.8, ls="--", label=f"nominal {level:.2f}")
    ax.set_title(r"(a) AR(1): coverage vs $\phi$ (pooled over $T$)")
    ax.set_xlabel(r"AR(1) coefficient $\phi$"); ax.set_ylabel("empirical coverage")
    ax.legend(fontsize=6, loc="lower left", ncol=2)

    ax = axes[1]
    rows = pd.DataFrame(results["coverage_vs_garch_persistence"])
    for m in DEGRADATION_METHODS:
        sub = rows[rows["method"] == m].sort_values("mean_persistence")
        err = np.vstack([sub["coverage"] - sub["wilson_lo"],
                         sub["wilson_hi"] - sub["coverage"]])
        ax.errorbar(sub["mean_persistence"], sub["coverage"], yerr=err, fmt="o-",
                    ms=3, lw=1.1, capsize=2, color=DEGRADATION_COLORS[m],
                    label=METHOD_LABELS[m])
    ax.axhline(level, color="k", lw=0.8, ls="--")
    ax.set_title(r"(b) GARCH(1,1): coverage vs persistence $\alpha+\beta$")
    ax.set_xlabel(r"persistence $\alpha+\beta$ (bin mean)")
    ax.set_ylabel("empirical coverage")
    fig.tight_layout(); fig.savefig(path); plt.close(fig)


# --------------------------------------------------------------------------- #
# Fig 4: width vs coverage tradeoff (dependent DGPs)
# --------------------------------------------------------------------------- #
def fig_width_coverage(path: Path, df: pd.DataFrame, level: float = 0.95) -> None:
    dep = df[df["cfg_family"].isin(DEPENDENT_FAMILIES)]
    t_values = sorted(dep["cfg_t_steps"].unique())
    fig, axes = plt.subplots(1, len(t_values), figsize=(11, 3.4), sharey=False)
    if len(t_values) == 1:
        axes = [axes]
    cmap = plt.cm.tab20(np.linspace(0, 1, len(METHOD_ORDER)))
    for ax, t in zip(axes, t_values):
        sub = dep[dep["cfg_t_steps"] == t]
        for color, m in zip(cmap, METHOD_ORDER):
            tag = f"{m}_{round(level * 100):d}"
            cov = sub[f"cov_{tag}"]
            n, k = int(cov.notna().sum()), int(cov.sum())
            coverage = k / n
            lo, hi = wilson_ci(k, n)
            width = float(sub[f"width_{tag}"].mean())
            ax.errorbar(coverage, width, xerr=[[coverage - lo], [hi - coverage]],
                        fmt="o", ms=4, capsize=2, color=color)
            ax.annotate(METHOD_LABELS[m], (coverage, width), fontsize=5.2,
                        xytext=(3, 3), textcoords="offset points")
        ax.axvline(level, color="k", lw=0.8, ls="--")
        ax.set_title(f"T = {int(t)} (dependent DGPs pooled)")
        ax.set_xlabel(f"empirical coverage ({level:.0%} nominal)")
    axes[0].set_ylabel("mean CI width (annualized Sharpe units)")
    fig.suptitle("Width vs coverage: an interval is only cheap if it still covers",
                 fontsize=9, y=1.02)
    fig.tight_layout(); fig.savefig(path); plt.close(fig)


# --------------------------------------------------------------------------- #
def main() -> None:
    FIGDIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(RESULTS / "records.csv")
    results = json.loads((RESULTS / "results.json").read_text())

    fig_setup(FIGDIR / "fig_setup.pdf")
    fig_coverage_heatmap(FIGDIR / "fig_coverage_heatmap.pdf", df)
    fig_degradation(FIGDIR / "fig_degradation.pdf", results)
    fig_width_coverage(FIGDIR / "fig_width_coverage.pdf", df)
    print(f"wrote 4 figures to {FIGDIR}")


if __name__ == "__main__":
    main()
