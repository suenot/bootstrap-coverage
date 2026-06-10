"""Reproduce every number and figure input in the paper.

    python scripts/run_all.py            # full run -> results/results.json + CSVs
    python scripts/run_all.py --quick    # small batch for a smoke check

Deterministic given the fixed seeds below: every experiment draws its RNG from
a SeedSequence-spawned child in a fixed order, so the numbers are identical for
any ``--jobs`` value (parallelism only changes wall time). Run from anywhere;
paths are anchored to the project root.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bootstrap_experiments import __version__
from bootstrap_experiments import analysis as A
from bootstrap_experiments.methods import METHOD_ORDER, method_parameters
from bootstrap_experiments.model import (
    DEPENDENT_FAMILIES,
    FAMILIES,
    T_GRID,
    sampling_ranges,
)
from bootstrap_experiments.simulate import MDD_T_GRID, run_batch, run_mdd_batch

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"

SEEDS = {"sharpe_batch": 20_260_610, "mdd_batch": 31_415, "mdd_truth": 7_777}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--quick", action="store_true", help="small smoke-check run")
    ap.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 2) - 2),
                    help="worker processes (does not affect the numbers)")
    ap.add_argument("--n-per-cell", type=int, default=None,
                    help="override experiments per (family, T) cell")
    ap.add_argument("--n-boot", type=int, default=None,
                    help="override bootstrap replications")
    args = ap.parse_args()

    if args.quick:
        n_per_cell = args.n_per_cell or 8
        n_boot = args.n_boot or 300
        mdd_per_cfg, mdd_truth_paths = 4, 3000
    else:
        n_per_cell = args.n_per_cell or 400
        n_boot = args.n_boot or 2000
        mdd_per_cfg, mdd_truth_paths = 150, 20_000

    RESULTS.mkdir(exist_ok=True)
    t_start = time.perf_counter()
    timings: dict[str, float] = {}

    n_total = n_per_cell * len(FAMILIES) * len(T_GRID)
    print(f"[1/4] Sharpe coverage batch: {n_total} experiments "
          f"({n_per_cell} per family x T cell, B={n_boot}, jobs={args.jobs}) ...",
          flush=True)
    t0 = time.perf_counter()
    recs = run_batch(n_per_cell, seed=SEEDS["sharpe_batch"], n_boot=n_boot,
                     n_jobs=args.jobs, progress_every=max(1, n_total // 8))
    timings["sharpe_batch_sec"] = time.perf_counter() - t0
    df = A.to_frame(recs)
    df.to_csv(RESULTS / "records.csv", index=False)
    print(f"  done in {timings['sharpe_batch_sec']:.1f}s -> records.csv", flush=True)

    n_mdd = mdd_per_cfg * len(FAMILIES) * len(MDD_T_GRID)
    print(f"[2/4] max-drawdown study: {n_mdd} experiments "
          f"(truth = {mdd_truth_paths} paths/config, cached) ...", flush=True)
    t0 = time.perf_counter()
    mdd_recs = run_mdd_batch(
        mdd_per_cfg, seed=SEEDS["mdd_batch"], truth_seed=SEEDS["mdd_truth"],
        n_truth_paths=mdd_truth_paths, n_boot=n_boot,
        cache_path=RESULTS / "mdd_truth_cache.npz", n_jobs=args.jobs,
        progress_every=max(1, n_mdd // 4))
    timings["mdd_batch_sec"] = time.perf_counter() - t0
    mdd_df = A.to_frame(mdd_recs)
    mdd_df.to_csv(RESULTS / "mdd_records.csv", index=False)
    print(f"  done in {timings['mdd_batch_sec']:.1f}s -> mdd_records.csv", flush=True)

    print("[3/4] summaries ...", flush=True)
    t0 = time.perf_counter()
    summary = A.summarize(df, mdd_df)
    timings["analysis_sec"] = time.perf_counter() - t0
    timings["total_sec"] = time.perf_counter() - t_start

    print("[4/4] writing results.json ...", flush=True)
    results = {
        "meta": {
            "package_version": __version__,
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "quick": bool(args.quick),
            "n_per_cell": n_per_cell,
            "n_boot": n_boot,
            "mdd_per_cfg": mdd_per_cfg,
            "mdd_truth_paths": mdd_truth_paths,
            "t_grid": T_GRID,
            "mdd_t_grid": MDD_T_GRID,
            "seeds": SEEDS,
            "n_jobs": args.jobs,
            "sampling_ranges": sampling_ranges(),
            "method_parameters": method_parameters(),
            "timings": {k: round(v, 2) for k, v in timings.items()},
            "notes": "Deterministic; identical numbers for any --jobs. "
                     "Reproduce with: python scripts/run_all.py",
        },
        **summary,
    }
    (RESULTS / "results.json").write_text(json.dumps(results, indent=2, default=float))
    print(f"\nWrote {RESULTS / 'results.json'}, records.csv, mdd_records.csv "
          f"(total {timings['total_sec']:.1f}s).")

    # ---- headline numbers to stdout ---------------------------------------
    print("\n--- HEADLINE: empirical coverage of the true Sharpe, 90% nominal ---")
    fam_short = {"iid_gaussian": "iidG", "iid_student_t": "iid-t", "ar1": "AR(1)",
                 "garch": "GARCH", "regime_switching": "regime"}
    for t in T_GRID:
        piv = A.coverage_pivot(df, 0.90, t=t)
        header = " ".join(f"{fam_short[f]:>7}" for f in piv.columns)
        print(f"\nT = {t:>5}          {header}")
        for m in METHOD_ORDER:
            row = " ".join(f"{piv.loc[m, f]:7.3f}" for f in piv.columns)
            print(f"  {m:<15} {row}")

    dep = df[df["cfg_family"].isin(DEPENDENT_FAMILIES)]
    print("\nDependent DGPs pooled (coverage @90 / @95, mean width @95):")
    for m in METHOD_ORDER:
        c90 = dep[f"cov_{m}_90"].mean()
        c95 = dep[f"cov_{m}_95"].mean()
        w95 = dep[f"width_{m}_95"].mean()
        print(f"  {m:<15} {c90:6.3f} / {c95:6.3f}   width {w95:6.3f}")

    cal = summary["iid_calibration"]
    print(f"\niid DGPs pooled ({cal['n_experiments']} experiments) -- "
          "the honest positive check, 95% nominal:")
    for m in METHOD_ORDER:
        c = cal["methods"][m]["level_95"]
        print(f"  {m:<15} coverage {c['coverage']:.3f} "
              f"[{c['wilson_lo']:.3f}, {c['wilson_hi']:.3f}]  -> {c['verdict']}")

    if "mdd" in summary:
        print("\nMax drawdown, 5% 'worst case' quantile -- mean realized tail prob "
              "F_true(q_hat) (nominal 0.05):")
        rows = pd.DataFrame(summary["mdd"]["summary"])
        sub = rows[rows["nominal_q"] == 0.05]
        piv = sub.pivot_table(index="method", columns=["family", "t"],
                              values="mean_realized_prob")
        with pd.option_context("display.width", 200, "display.precision", 3):
            print(piv)


if __name__ == "__main__":
    main()
