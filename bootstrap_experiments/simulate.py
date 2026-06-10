"""One experiment = one simulated return stream, every CI method applied.

Sharpe study: draw a random DGP config (family + sampled parameters), generate
one length-T sample, build 90% and 95% CIs for the annualized Sharpe with every
method, and record whether the TRUE (analytically known) Sharpe is covered,
plus the interval width.

Max-drawdown study: for a fixed grid of canonical configs, the TRUE sampling
distribution of the length-T maximum drawdown is estimated once by
mega-simulation (cached to disk, seeded). Each experiment then asks every
bootstrap method to predict the q-quantile of that distribution and we record
the *realized* tail probability ``F_true(q_hat)`` -- for a calibrated method
its average equals q.

Reproducibility: every experiment gets its own ``SeedSequence``-spawned child
RNG in a fixed order, so results are bit-identical regardless of the number of
worker processes (``n_jobs`` only changes wall time, never numbers).
"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from .methods import (
    DEFAULT_LEVELS,
    DEFAULT_N_BOOT,
    MDD_QS,
    max_drawdown_paths,
    mdd_bootstrap_quantiles,
    sharpe_annualized,
    sharpe_ci_bundle,
)
from .model import (
    FAMILIES,
    T_GRID,
    DGPConfig,
    canonical_configs,
    generate,
    generate_paths,
    sample_config,
)

MDD_T_GRID: tuple[int, ...] = (250, 1000)  # path lengths for the max-drawdown study


# --------------------------------------------------------------------------- #
# Sharpe-coverage experiments
# --------------------------------------------------------------------------- #
def run_experiment(cfg: DGPConfig, rng: np.random.Generator,
                   n_boot: int = DEFAULT_N_BOOT,
                   levels: tuple[float, ...] = DEFAULT_LEVELS) -> dict:
    """Simulate one sample, build all CIs, score coverage of the true Sharpe."""
    r = generate(cfg, rng)
    truth = cfg.true_sharpe_annual
    bundle = sharpe_ci_bundle(r, rng, n_boot=n_boot, levels=levels)

    rec = cfg.as_record()
    rec["sharpe_hat"] = sharpe_annualized(r)
    for name, m in bundle.items():
        for lvl, (lo, hi) in m["ci"].items():
            tag = f"{name}_{round(lvl * 100):d}"
            rec[f"cov_{tag}"] = int(lo <= truth <= hi)
            rec[f"width_{tag}"] = hi - lo
        for key, val in m["extra"].items():
            rec[f"x_{name}_{key}"] = val
    return rec


def _sharpe_task(args: tuple) -> dict:
    family, t_steps, n_boot, levels, child_ss = args
    rng = np.random.default_rng(child_ss)
    cfg = sample_config(rng, family, t_steps)
    return run_experiment(cfg, rng, n_boot=n_boot, levels=levels)


def run_batch(n_per_cell: int,
              *,
              families: tuple[str, ...] = FAMILIES,
              t_grid: tuple[int, ...] = T_GRID,
              seed: int = 0,
              n_boot: int = DEFAULT_N_BOOT,
              levels: tuple[float, ...] = DEFAULT_LEVELS,
              n_jobs: int = 1,
              progress_every: int = 0) -> list[dict]:
    """``n_per_cell`` experiments for every (family, T) cell, reproducibly.

    Child seeds are spawned in a fixed order before any work is dispatched, so
    the records are identical for any ``n_jobs``.
    """
    cells = [(fam, t) for fam in families for t in t_grid]
    tasks = []
    ss = np.random.SeedSequence(seed)
    children = ss.spawn(len(cells) * n_per_cell)
    i = 0
    for fam, t in cells:
        for _ in range(n_per_cell):
            tasks.append((fam, t, n_boot, levels, children[i]))
            i += 1

    records: list[dict] = []
    if n_jobs <= 1:
        for j, task in enumerate(tasks):
            records.append(_sharpe_task(task))
            if progress_every and (j + 1) % progress_every == 0:
                print(f"  {j + 1}/{len(tasks)}", flush=True)
    else:
        chunk = max(1, len(tasks) // (n_jobs * 16))
        with ProcessPoolExecutor(max_workers=n_jobs) as pool:
            for j, rec in enumerate(pool.map(_sharpe_task, tasks, chunksize=chunk)):
                records.append(rec)
                if progress_every and (j + 1) % progress_every == 0:
                    print(f"  {j + 1}/{len(tasks)}", flush=True)
    return records


# --------------------------------------------------------------------------- #
# max-drawdown truth cache (mega-simulation per canonical config)
# --------------------------------------------------------------------------- #
def _mdd_cache_key(label: str, t_steps: int, n_paths: int, seed: int) -> str:
    return f"{label}_T{t_steps}_n{n_paths}_s{seed}"


def mdd_truth_distribution(cfg: DGPConfig, n_paths: int, seed: int,
                           chunk: int = 4000) -> np.ndarray:
    """Sorted sample (size ``n_paths``) from the TRUE max-drawdown distribution
    of length-T paths of this DGP, by direct mega-simulation."""
    rng = np.random.default_rng(np.random.SeedSequence(seed))
    parts = []
    done = 0
    while done < n_paths:
        m = min(chunk, n_paths - done)
        parts.append(max_drawdown_paths(generate_paths(cfg, rng, n_paths=m)))
        done += m
    return np.sort(np.concatenate(parts))


def ensure_mdd_truth_cache(configs: dict[str, DGPConfig], cache_path: Path,
                           n_paths: int, seed: int) -> dict[str, np.ndarray]:
    """Load (or compute and persist) the truth distributions for ``configs``.

    Keys embed (label, T, n_paths, seed), so quick and full runs never collide.
    """
    cache: dict[str, np.ndarray] = {}
    if cache_path.exists():
        with np.load(cache_path) as z:
            cache = {k: z[k] for k in z.files}
    missing = {}
    for name, cfg in configs.items():
        key = _mdd_cache_key(cfg.label, cfg.t_steps, n_paths, seed)
        if key not in cache:
            missing[key] = cfg
    for key, cfg in missing.items():
        print(f"  mega-sim truth: {key}", flush=True)
        cache[key] = mdd_truth_distribution(cfg, n_paths=n_paths, seed=seed)
    if missing:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(cache_path, **cache)
    return {
        _mdd_cache_key(cfg.label, cfg.t_steps, n_paths, seed): cache[
            _mdd_cache_key(cfg.label, cfg.t_steps, n_paths, seed)]
        for cfg in configs.values()
    }


# --------------------------------------------------------------------------- #
# max-drawdown experiments
# --------------------------------------------------------------------------- #
def run_mdd_experiment(cfg: DGPConfig, rng: np.random.Generator,
                       truth_sorted: np.ndarray,
                       n_boot: int = DEFAULT_N_BOOT,
                       qs: tuple[float, ...] = MDD_QS) -> dict:
    """One sample; every method predicts MDD q-quantiles; score vs the truth.

    Records, per method and q: the realized probability
    ``F_true(q_hat) = P_true(MDD <= q_hat)`` (mean over experiments should be q
    for a calibrated method) and the signed relative error of ``q_hat`` against
    the true q-quantile.
    """
    r = generate(cfg, rng)
    bundle = mdd_bootstrap_quantiles(r, rng, n_boot=n_boot, qs=qs)
    n_truth = truth_sorted.size

    rec = cfg.as_record()
    rec["mdd_hat"] = float(max_drawdown_paths(r[None, :])[0])
    for q in qs:
        rec[f"true_q{round(q * 100):d}"] = float(np.quantile(truth_sorted, q))
    for name, m in bundle.items():
        for q, q_hat in m["q"].items():
            tag = f"{name}_q{round(q * 100):d}"
            realized = float(np.searchsorted(truth_sorted, q_hat, side="right")) / n_truth
            true_q = float(np.quantile(truth_sorted, q))
            rec[f"prob_{tag}"] = realized
            rec[f"qhat_{tag}"] = q_hat
            rec[f"relerr_{tag}"] = (q_hat - true_q) / abs(true_q)
    return rec


def _mdd_task(args: tuple) -> dict:
    cfg, truth_sorted, n_boot, qs, child_ss = args
    rng = np.random.default_rng(child_ss)
    return run_mdd_experiment(cfg, rng, truth_sorted, n_boot=n_boot, qs=qs)


def run_mdd_batch(n_per_cfg: int,
                  *,
                  t_grid: tuple[int, ...] = MDD_T_GRID,
                  seed: int = 0,
                  truth_seed: int = 7_777,
                  n_truth_paths: int = 20_000,
                  n_boot: int = DEFAULT_N_BOOT,
                  qs: tuple[float, ...] = MDD_QS,
                  cache_path: Path | None = None,
                  n_jobs: int = 1,
                  progress_every: int = 0) -> list[dict]:
    """Max-drawdown study over the canonical config grid (families x T)."""
    configs: dict[str, DGPConfig] = {}
    for t in t_grid:
        for name, cfg in canonical_configs(t_steps=t).items():
            configs[f"{name}_T{t}"] = cfg
    if cache_path is None:
        cache_path = Path("results") / "mdd_truth_cache.npz"
    truths = ensure_mdd_truth_cache(configs, cache_path, n_paths=n_truth_paths,
                                    seed=truth_seed)

    ss = np.random.SeedSequence(seed)
    children = ss.spawn(len(configs) * n_per_cfg)
    tasks = []
    i = 0
    for cfg in configs.values():
        key = _mdd_cache_key(cfg.label, cfg.t_steps, n_truth_paths, truth_seed)
        for _ in range(n_per_cfg):
            tasks.append((cfg, truths[key], n_boot, qs, children[i]))
            i += 1

    records: list[dict] = []
    if n_jobs <= 1:
        for j, task in enumerate(tasks):
            records.append(_mdd_task(task))
            if progress_every and (j + 1) % progress_every == 0:
                print(f"  {j + 1}/{len(tasks)}", flush=True)
    else:
        chunk = max(1, len(tasks) // (n_jobs * 16))
        with ProcessPoolExecutor(max_workers=n_jobs) as pool:
            for j, rec in enumerate(pool.map(_mdd_task, tasks, chunksize=chunk)):
                records.append(rec)
                if progress_every and (j + 1) % progress_every == 0:
                    print(f"  {j + 1}/{len(tasks)}", flush=True)
    return records
