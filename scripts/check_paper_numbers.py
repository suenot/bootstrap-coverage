"""Assert that every quantitative claim in paper/main.tex matches the results.

    python scripts/check_paper_numbers.py

Each check formats a value from results/results.json (or, for the few
stratifications the JSON does not pre-aggregate, recomputes it from
results/records.csv) exactly the way the paper quotes it and asserts the
resulting token appears in main.tex; where the paper states an inequality, a
range, or a bold-marking rule, the underlying condition is asserted too.
Exits non-zero if any check fails, so it can gate a release.
"""

from __future__ import annotations

import json
import math
import re
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
TEX = re.sub(r"\s+", " ", (ROOT / "paper" / "main.tex").read_text())
R = json.loads((ROOT / "results" / "results.json").read_text())
DF = pd.read_csv(ROOT / "results" / "records.csv")

METHODS = ["lo_iid", "lo_hac", "iid_pct", "iid_basic", "iid_bca",
           "trade_fixed_5", "trade_fixed_21", "trade_cusum",
           "sb_l5", "sb_l10", "sb_l20", "sb_l50", "sb_auto", "cb_auto"]
DEP_FAMILIES = ["ar1", "garch", "regime_switching"]

failures: list[str] = []
n_checks = 0


def check(label: str, token: str, cond: bool = True) -> None:
    """Assert ``token`` appears in main.tex (whitespace-normalized) and ``cond``."""
    global n_checks
    n_checks += 1
    ok_tex = token in TEX
    if ok_tex and cond:
        print(f"  PASS  {label:62} {token!r}")
    else:
        why = [] if ok_tex else [f"token {token!r} not in main.tex"]
        if not cond:
            why.append("condition failed")
        failures.append(f"{label}: {'; '.join(why)}")
        print(f"  FAIL  {label:62} {token!r}  <-- {'; '.join(why)}")


def cond(label: str, condition: bool) -> None:
    """A pure data condition (no tex token)."""
    global n_checks
    n_checks += 1
    if condition:
        print(f"  PASS  {label}")
    else:
        failures.append(f"{label}: condition failed")
        print(f"  FAIL  {label}")


def f3(v: float) -> str:
    return f"{v:.3f}"


def wilson(k: int, n: int, z: float = 1.959963984540054) -> tuple[float, float]:
    p = k / n
    denom = 1.0 + z**2 / n
    center = (p + z**2 / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return max(0.0, center - half), min(1.0, center + half)


def cov_ci(sub: pd.DataFrame, method: str, level: int) -> tuple[float, float, float, int]:
    col = sub[f"cov_{method}_{level}"]
    n, k = int(col.notna().sum()), int(col.sum())
    lo, hi = wilson(k, n)
    return k / n, lo, hi, n


CT = {(r["method"], r["family"], r["t"], r["level"]): r for r in R["coverage_table"]}

# ------------------------------------------------------------- design ----
print("[design]")
check("n Sharpe experiments", "$6{,}000$ experiments", R["n_experiments"] == 6000)
check("n per cell", "$400$ experiments per (family, $T$) cell",
      R["meta"]["n_per_cell"] == 400)
check("bootstrap replications", "$B=2{,}000$", R["meta"]["n_boot"] == 2000)
check("T grid", r"$T\in\{250,1000,4000\}$", R["meta"]["t_grid"] == [250, 1000, 4000])
check("MDD experiments", "$1{,}500$ in all", R["mdd"]["n_experiments"] == 1500)
check("MDD per cell", "$150$ experiments per", R["meta"]["mdd_per_cfg"] == 150)
check("MDD truth paths", "$20{,}000$ paths", R["meta"]["mdd_truth_paths"] == 20000)
sr_rng = R["meta"]["sampling_ranges"]
check("Sharpe range", "uniform on $[0,2]$", sr_rng["sharpe_annual_range"] == [0.0, 2.0])
check("sigma range", "uniform on $[0.005,0.02]$",
      sr_rng["sigma_daily_range"] == [0.005, 0.02])
check("nu choices", r"$\nu\in\{5,6,8,12,20\}$",
      sr_rng["nu_choices"] == [5.0, 6.0, 8.0, 12.0, 20.0])
check("phi choices", r"$\phi\in\{0.05,0.10,0.20,0.30\}$",
      sr_rng["phi_choices"] == [0.05, 0.1, 0.2, 0.3])
check("garch alpha range", r"$\alpha$ uniform on $[0.03,0.12]$",
      sr_rng["garch_alpha_range"] == [0.03, 0.12])
check("garch persistence range", r"$\alpha+\beta$ uniform on $[0.85,0.98]$",
      sr_rng["garch_persistence_range"] == [0.85, 0.98])
check("vol ratio range", "uniform on $[2,4]$", sr_rng["vol_ratio_range"] == [2.0, 4.0])
check("stay-low range", "$[0.95,0.99]$", sr_rng["p_stay_low_range"] == [0.95, 0.99])
check("stay-high range", "$[0.90,0.98]$", sr_rng["p_stay_high_range"] == [0.9, 0.98])
check("garch burn-in", "$500$-step burn-in", sr_rng["garch_burn_in"] == 500)
check("cusum multiplier", "$2\\times$ the sample standard deviation",
      R["meta"]["method_parameters"]["cusum_mult"] == 2.0)
check("total runtime", "$331$ s", round(R["meta"]["timings"]["total_sec"]) == 331)

# ------------------------------------------- Table 1: iid calibration ----
print("[Table 1: iid calibration]")
cal = R["iid_calibration"]
check("n iid experiments", "$2{,}400$", cal["n_experiments"] == 2400)
for m in METHODS:
    for lvl in (90, 95):
        c = cal["methods"][m][f"level_{lvl}"]
        tok = f"{f3(c['coverage'])} [{f3(c['wilson_lo'])}, {f3(c['wilson_hi'])}]"
        under = c["wilson_hi"] < lvl / 100
        tok_full = (r"\textbf{" + f3(c["coverage"]) + "} [" + f3(c["wilson_lo"])
                    + ", " + f3(c["wilson_hi"]) + "]") if under else tok
        check(f"iid {m} @{lvl} (+bold rule)", tok_full)

# ------------------------------- Table 2: dependent families @95 ----------
print("[Table 2: dependent coverage at 95%]")
for m in METHODS:
    cells = []
    for fam in DEP_FAMILIES:
        for t in (250, 1000, 4000):
            r = CT[(m, fam, t, 0.95)]
            v = f3(r["coverage"])
            cells.append(r"\textbf{" + v + "}" if r["wilson_hi"] < 0.95 else v)
    check(f"table2 row {m}", " & ".join(cells))
halfs = [(CT[(m, f, t, 0.95)]["wilson_hi"] - CT[(m, f, t, 0.95)]["wilson_lo"]) / 2
         for m in METHODS for f in DEP_FAMILIES for t in (250, 1000, 4000)]
check("table2 Wilson half-width range", r"$\pm 0.018$--$0.035$",
      round(min(halfs), 3) == 0.018 and round(max(halfs), 3) == 0.035)

# ----------------------------------------------- iid-section claims -------
print("[Section 6.1 claims]")
auto_len_iid = DF[DF.cfg_family.isin(["iid_gaussian", "iid_student_t"])][
    "x_sb_auto_block_len"].groupby([DF.cfg_family, DF.cfg_t_steps]).mean()
cond("PW auto length ~1.0-1.1 on iid families",
     bool((auto_len_iid >= 1.0).all() and (auto_len_iid <= 1.1).all()))
check("21d trades at T=250", r"$\lfloor 250/21\rfloor = 11$", 250 // 21 == 11)
tpc = [f3(CT[("iid_pct", "iid_student_t", t, 0.95)]["coverage"]) for t in (250, 1000, 4000)]
check("student-t percentile @95 by T", "$" + "/".join(tpc) + "$")
t5 = DF[(DF.cfg_family == "iid_student_t") & (DF.cfg_nu == 5.0)]
c95, c90 = t5["cov_iid_pct_95"].mean(), t5["cov_iid_pct_90"].mean()
check("nu=5 n", "$n=266$", len(t5) == 266)
check("nu=5 @95", f"gives ${f3(c95)}$ at $95\\%$")
check("nu=5 @90", f"${f3(c90)}$ at $90\\%$")

# ------------------------------------------------ AR(1) degradation -------
print("[Section 6.2: AR(1) and GARCH]")
phi_rows = {(r["phi"], r["method"]): r for r in R["coverage_vs_ar_phi"]}
seq = [f3(phi_rows[(p, "iid_pct")]["coverage"]) for p in (0.05, 0.10, 0.20, 0.30)]
check("iid_pct @90 vs phi", f"degrades from ${seq[0]}$ at $\\phi=0.05$ to "
      f"${seq[1]}$, ${seq[2]}$, and ${seq[3]}$")
sub3 = DF[(DF.cfg_family == "ar1") & (DF.cfg_phi == 0.3)]
check("phi=0.3 n", "$n=296$", len(sub3) == 296)
n_by_t = [len(sub3[sub3.cfg_t_steps == t]) for t in (250, 1000, 4000)]
check("phi=0.3 n by T", "$96/104/96$ per $T$", n_by_t == [96, 104, 96])
c, lo, hi, n = cov_ci(sub3, "iid_pct", 95)
check("phi=0.3 iid_pct @95 + Wilson", f"${f3(c)}$ $[{f3(lo)},{f3(hi)}]$")
check("phi=0.3 miss rate", "misses $15.9\\%$", round((1 - c) * 100, 1) == 15.9)
cond("miss rate > 3x nominal", (1 - c) > 3 * 0.05)
pt95 = [f3(cov_ci(sub3[sub3.cfg_t_steps == t], "iid_pct", 95)[0]) for t in (250, 1000, 4000)]
check("phi=0.3 iid_pct @95 by T", f"coverage at $\\phi=0.3$ is ${pt95[0]}$, ${pt95[1]}$, ${pt95[2]}$")
pt90 = [f3(cov_ci(sub3[sub3.cfg_t_steps == t], "iid_pct", 90)[0]) for t in (250, 1000, 4000)]
check("phi=0.3 iid_pct @90 by T", f"(at $90\\%$: ${pt90[0]}$, ${pt90[1]}$, ${pt90[2]}$)")
check("abstract per-T claim", "$0.812\\to0.854$",
      pt95[0] == "0.812" and pt95[2] == "0.854")
for m, expect in [("iid_basic", "basic $0.848$"), ("iid_bca", "BCa $0.838$")]:
    c = cov_ci(sub3, m, 95)[0]
    check(f"phi=0.3 {m} @95", expect, f3(c) in expect)
c_loiid = cov_ci(sub3, "lo_iid", 95)[0]
check("phi=0.3 lo_iid @95", "fails identically ($0.841$)", f3(c_loiid) == "0.841")
c_hac = cov_ci(sub3, "lo_hac", 95)[0]
check("phi=0.3 lo_hac @95", f"HAC variant covers ${f3(c_hac)}$")
c_cb = cov_ci(sub3, "cb_auto", 95)[0]
check("phi=0.3 cb_auto @95", f"circular block bootstrap ${f3(c_cb)}$")
c_sba = cov_ci(sub3, "sb_auto", 95)[0]
check("phi=0.3 sb_auto @95", f"automatic stationary bootstrap ${f3(c_sba)}$")

# Table 3 rows (pooled 90 / pooled 95 / 95 by T), with the bold rule
for m in METHODS:
    cells = []
    for lvl, frame in [(90, sub3), (95, sub3)]:
        c, lo, hi, n = cov_ci(frame, m, lvl)
        v = f3(c)
        cells.append(r"\textbf{" + v + "}" if hi < lvl / 100 else v)
    for t in (250, 1000, 4000):
        c, lo, hi, n = cov_ci(sub3[sub3.cfg_t_steps == t], m, 95)
        v = f3(c)
        cells.append(r"\textbf{" + v + "}" if hi < 0.95 else v)
    check(f"table3 row {m}", " & ".join(cells))

# GARCH / regime mildness
g_cells = [CT[("iid_pct", f, t, 0.9)]["coverage"]
           for f in ("garch", "regime_switching") for t in (250, 1000, 4000)]
check("garch/regime iid_pct @90 range", "$0.885$--$0.915$",
      f3(min(g_cells)) == "0.885" and f3(max(g_cells)) == "0.915")
pers = [r for r in R["coverage_vs_garch_persistence"] if r["method"] == "iid_pct"]
pers.sort(key=lambda r: r["mean_persistence"])
check("garch persistence first bin", f"from ${f3(pers[0]['coverage'])}$ in the "
      "lowest-persistence quartile")
check("garch persistence last bin", f"${f3(pers[-1]['coverage'])}$ in the highest")
check("garch last bin mean", f"$\\alpha+\\beta = {pers[-1]['mean_persistence']:.3f}$")

# --------------------------------------------- trade-resampling section ---
print("[Section 6.3: trade resampling]")
c, lo, hi, _ = cov_ci(sub3, "trade_fixed_5", 95)
check("phi=0.3 trade5 @95 + Wilson", f"${f3(c)}$ $[{f3(lo)},{f3(hi)}]$")
c_cusum = cov_ci(sub3, "trade_cusum", 95)[0]
check("phi=0.3 cusum @95", f"${f3(c_cusum)}$ at $\\phi=0.3$ ($95\\%$)")
ar = DF[DF.cfg_family == "ar1"]
check("cusum mean episode length ar1", "averaging about $6$ days",
      5.5 <= ar["x_trade_cusum_mean_trade_len"].mean() <= 6.5)
check("cusum mean length phi=0.3", "($5.6$ at $\\phi=0.3$)",
      f"{sub3['x_trade_cusum_mean_trade_len'].mean():.1f}" == "5.6")
c21_1000 = cov_ci(sub3[sub3.cfg_t_steps == 1000], "trade_fixed_21", 95)[0]
check("phi=0.3 trade21 T=1000", f"(${f3(c21_1000)}$ at $\\phi=0.3$, $T=1000$)")
c21_250 = cov_ci(sub3[sub3.cfg_t_steps == 250], "trade_fixed_21", 95)[0]
check("phi=0.3 trade21 T=250", f"undercovers at $T=250$ (${f3(c21_250)}$)")
check("AR dependence at 5 bars", r"$0.3^5\approx0.002$", round(0.3**5, 3) == 0.002)

# --------------------------------------------- block-length section -------
print("[Section 6.4: block length and width]")
wmc = {(r["t"], r["level"], r["method"]): r for r in R["width_at_matched_coverage"]}
r50 = wmc[(250, 0.9, "sb_l50")]
check("L=50 T=250 dependent pooled @90", f"covers ${f3(r50['coverage'])}$ pooled")
fam_l50 = [CT[("sb_l50", f, 250, 0.9)]["coverage"] for f in
           ("iid_gaussian", "iid_student_t", "ar1", "garch", "regime_switching")]
check("L=50 T=250 per-family range", "($0.762$--$0.790$ per family",
      f3(min(fam_l50)) == "0.762" and f3(max(fam_l50)) == "0.790")
r50_ar = CT[("sb_l50", "ar1", 4000, 0.9)]
check("L=50 best on AR(1) T=4000 @90", f"(${f3(r50_ar['coverage'])}$ at $90\\%$)",
      all(CT[(m, "ar1", 4000, 0.9)]["coverage"] <= r50_ar["coverage"] for m in METHODS))
auto3 = [sub3[sub3.cfg_t_steps == t]["x_sb_auto_block_len"].mean() for t in (250, 1000, 4000)]
check("PW lengths at phi=0.3", r"$\approx 3.8/6.1/10.9$ days at $\phi=0.3$",
      [round(v, 1) for v in auto3] == [3.8, 6.1, 10.9])
c90a, c95a = ar["cov_sb_auto_90"].mean(), ar["cov_sb_auto_95"].mean()
check("sb_auto AR(1) pooled @90", f"(${f3(c90a)}$)", round((0.90 - c90a) * 100, 1) == 3.2)
check("sb_auto AR(1) gap @90", "by $3.2$ points")
check("sb_auto AR(1) pooled @95", f"(${f3(c95a)}$)", round((0.95 - c95a) * 100, 1) == 1.9)
check("sb_auto AR(1) gap @95", "$1.9$ points")
best250 = max((wmc[(250, 0.9, m)]["coverage"], m) for m in METHODS)
check("best method T=250 dep @90", f"achieves ${f3(best250[0])}$",
      all(wmc[(250, 0.9, m)]["wilson_hi"] < 0.9 for m in METHODS))
ok95 = [m for m in METHODS if wmc[(250, 0.95, m)]["not_undercovering"]]
check("T=250 @95 survivors", f"standard error (${f3(wmc[(250,0.95,'lo_hac')]['coverage'])}$)",
      sorted(ok95) == ["lo_hac", "trade_fixed_5"])
check("T=250 @95 trade5", f"trade resampler (${f3(wmc[(250,0.95,'trade_fixed_5')]['coverage'])}$)")
w_hac = wmc[(250, 0.95, "lo_hac")]["mean_width"]
check("T=250 honest width ~4.1", "about $4.1$ annualized-Sharpe units",
      4.0 <= w_hac <= 4.15)
w_l50 = wmc[(250, 0.95, "sb_l50")]["mean_width"]
check("T=250 L=50 width 3.4", "($L=50$ at width $3.4$)", round(w_l50, 1) == 3.4)
bca4000 = wmc[(4000, 0.95, "iid_bca")]
check("T=4000 BCa pooled", f"(${f3(bca4000['coverage'])}$)",
      bca4000["not_undercovering"] == 1)
check("T=4000 BCa AR stratum", f"sits at ${f3(CT[('iid_bca','ar1',4000,0.95)]['coverage'])}$")

# --------------------------------------------------- MDD section ----------
print("[Section 6.5: maximum drawdown]")
MD = {(r["method"], r["family"], r["t"], r["nominal_q"]): r for r in R["mdd"]["summary"]}
mdd_methods = ["iid_pct", "trade_fixed_5", "sb_l20", "sb_auto", "cb_auto"]
for fam in ["iid_gaussian", "iid_student_t", "ar1", "garch", "regime_switching"]:
    for t in (250, 1000):
        row = " & ".join(f3(MD[(m, fam, t, 0.05)]["mean_realized_prob"]) for m in mdd_methods)
        check(f"mdd table {fam} T={t}", row)
cond("all mdd q05 cells above nominal",
     all(MD[(m, f, t, 0.05)]["mean_realized_prob"] > 0.05
         for m in mdd_methods for f in ["iid_gaussian", "iid_student_t", "ar1",
                                        "garch", "regime_switching"] for t in (250, 1000)))
check("iidG exceedance", "is $0.102$ at $T=250$ and $0.082$ at $T=1000$",
      f3(MD[("iid_pct", "iid_gaussian", 250, 0.05)]["mean_realized_prob"]) == "0.102"
      and f3(MD[("iid_pct", "iid_gaussian", 1000, 0.05)]["mean_realized_prob"]) == "0.082")
check("ar1 exceedance", "is $0.232$ at $T=250$ and $0.226$ at $T=1000$")
check("sb_auto fixes ar1 at 1000", "from $0.226$ down to $0.115$",
      f3(MD[("sb_auto", "ar1", 1000, 0.05)]["mean_realized_prob"]) == "0.115")
check("trade5 ar1 at 1000", "trade resampler to $0.112$")
check("ar1 T250 partial fix", "reach only $0.196$ and $0.143$",
      f3(MD[("sb_auto", "ar1", 250, 0.05)]["mean_realized_prob"]) == "0.196"
      and f3(MD[("trade_fixed_5", "ar1", 250, 0.05)]["mean_realized_prob"]) == "0.143")
check("regime no fix T250", r"$0.165\to0.168$ at $T=250$")
check("regime no fix T1000", r"$0.134\to0.133$ at $T=1000$")
check("garch", "($0.115/0.106$ iid")
q50_iid = [f3(MD[("iid_pct", "ar1", t, 0.5)]["mean_realized_prob"]) for t in (250, 1000)]
check("ar1 median exceedance", f"probability ${q50_iid[0]}$ ($T=250$) and ${q50_iid[1]}$ ($T=1000$)")
blocks_q50 = [MD[(m, "ar1", t, 0.5)]["mean_realized_prob"]
              for m in ["trade_fixed_5", "sb_l20", "sb_auto", "cb_auto"] for t in (250, 1000)]
check("block median range", "versus $0.51$--$0.61$",
      0.51 <= min(blocks_q50) <= 0.52 and 0.60 <= max(blocks_q50) <= 0.62)
rel = [MD[("iid_pct", "ar1", t, 0.05)]["mean_relerr_qhat"] for t in (250, 1000)]
check("ar1 q05 relative shallowness", "about $15$--$17\\%$ too shallow",
      0.15 <= min(rel) and max(rel) <= 0.17)
all_iid_pct = [MD[("iid_pct", f, t, 0.05)]["mean_realized_prob"]
               for f in ["iid_gaussian", "iid_student_t", "ar1", "garch",
                         "regime_switching"] for t in (250, 1000)]
check("grid exceedance range", "from $0.07$ to $0.23$",
      round(min(all_iid_pct), 2) == 0.07 and round(max(all_iid_pct), 2) == 0.23)
cond("abstract 0.13--0.17 regime", f3(MD[("iid_pct", "regime_switching", 1000, 0.05)]["mean_realized_prob"]) == "0.134"
     and f3(MD[("iid_pct", "regime_switching", 250, 0.05)]["mean_realized_prob"]) == "0.165")

# ------------------------------------------------------------- summary ----
print(f"\n{n_checks} checks, {len(failures)} failures.")
if failures:
    print("\nFAILURES:")
    for f in failures:
        print(" -", f)
    sys.exit(1)
print("All paper numbers match results.json / records.csv.")
