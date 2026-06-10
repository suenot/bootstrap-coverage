# Do Bootstrap Confidence Intervals for Backtest Statistics Actually Cover?

A reproducible experiment harness behind a short methods paper that puts the
popular **"bootstrap your backtest trades to get confidence intervals"** advice
to a controlled test. With known data-generating processes the true annualized
Sharpe ratio is known *exactly* (and the true sampling distribution of the
maximum drawdown is estimable to arbitrary precision), so CI **coverage is
measurable exactly** -- by method x DGP x sample size.

Methods compared on the same simulated samples:

* iid bootstrap of returns (percentile / basic / BCa),
* **trade-level iid resampling** (the blog recipe: fixed-length and
  threshold-crossing trade episodes, resampled with replacement),
* **stationary bootstrap** (Politis-Romano; mean block length swept *and*
  Politis-White 2004 / Patton-Politis-White 2009 automatic block length),
* circular block bootstrap (automatic block length),
* analytic: Lo (2002) iid SE and its HAC autocorrelation-adjusted variant.

DGP families (all with analytically pinned unconditional mean/vol, hence exact
true Sharpe): iid Gaussian, iid Student-t, AR(1) (phi in {0.05,...,0.3}),
GARCH(1,1) (sampled realistic persistence), 2-state Markov regime-switching
volatility. T in {250, 1000, 4000}. Every parameter of every experiment is
sampled and recorded -- no hidden constants.

This grew out of a [marketmaker.cc](https://marketmaker.cc) blog post on Monte
Carlo bootstrap for backtests; the paper is the honest validation the post
never had (including the regimes where the post's method is perfectly fine).

## Reproduce everything

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python scripts/run_all.py            # full run -> results/results.json + CSVs
python -m bootstrap_experiments.figures   # -> paper/figures/*.pdf
```

Everything is deterministic given the seeds in `scripts/run_all.py`; the
worker count (`--jobs`) only changes wall time, never numbers. `--quick` runs
a small batch for a smoke check.

## Layout

```
bootstrap_experiments/
  model.py       # 5 return DGPs with exactly known annualized Sharpe
  methods.py     # CI constructors: iid/trade/stationary/circular bootstrap, Lo SEs
  simulate.py    # one experiment end-to-end, Monte-Carlo batches, MDD truth cache
  analysis.py    # coverage tables (Wilson CIs), degradation curves, width tradeoff
  figures.py     # the paper's 4 vector-PDF figures
scripts/run_all.py
tests/           # pytest sanity checks (python -m pytest -q)
results/         # results.json + records CSVs + MDD truth cache (generated)
```

## Tests

```bash
python -m pytest -q     # DGP moments/Sharpe identities, bootstrap machinery,
                        # iid calibration sanity, Lo-vs-bootstrap SE, determinism
```

## License

Code: [MIT](LICENSE). Paper text and figures: CC BY 4.0.
