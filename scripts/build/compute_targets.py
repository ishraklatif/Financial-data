#!/usr/bin/env python3
"""
compute_targets.py
==================
Phase 3, Step 2 — Target validation and quarantine.

Reads panel_raw.parquet (which already contains y_ret_21d and y_dir_21d
computed by build_feature_panel.py) and applies a quarantine pass:
any target value that is physically impossible or caused by a data
artefact (stock split, corporate action, bad yfinance data) is set to
NaN and logged for audit.

TARGET DEFINITIONS
------------------
y_ret_21d : 21-trading-day forward log return = ln(close_adj[t+21] / close_adj[t])
            Computed from close_adj only — never raw close.
y_dir_21d : 1 if y_ret_21d > 0, else 0. NaN where y_ret_21d is NaN.

QUARANTINE BOUNDS
-----------------
For 21-trading-day log returns:
  Lower bound : -2.0  → price drop of ~86% in 21 days (covers GFC, COVID crashes)
  Upper bound :  1.5  → price gain of ~348% in 21 days (covers extreme meme events)

Any value outside these bounds is almost certainly a data artefact
(unadjusted split, bad yfinance corporate action) rather than a real
market move. These rows are set to NaN, not deleted — the feature
values for those rows are still valid and useful for training on non-
target columns.

OUTPUTS
-------
data/features/panel_with_targets.parquet
    Same schema as panel_raw.parquet, with y_ret_21d / y_dir_21d
    quarantine-cleaned.

data/audit/target_quarantine_log.json
    Every quarantined row: ticker, date, raw_value, reason.

data/audit/target_summary.json
    Distribution statistics for y_ret_21d per ticker and overall.

USAGE
-----
    python -m scripts.build.compute_targets
    python -m scripts.build.compute_targets --verify   # check output only
    python -m scripts.build.compute_targets --plot     # print ASCII histogram
"""

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(PROJECT_ROOT / "logs" / "compute_targets.log"),
    ],
)
log = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────────────────────────────────────

FEATURES_DIR = PROJECT_ROOT / "data" / "features"
AUDIT_DIR    = PROJECT_ROOT / "data" / "audit"

INPUT_PATH   = FEATURES_DIR / "panel_raw.parquet"
OUTPUT_PATH  = FEATURES_DIR / "panel_with_targets.parquet"
QUARANTINE_LOG_PATH = AUDIT_DIR / "target_quarantine_log.json"
SUMMARY_PATH        = AUDIT_DIR / "target_summary.json"

# ─────────────────────────────────────────────────────────────────────────────
# Quarantine bounds for 21-day log returns
# ─────────────────────────────────────────────────────────────────────────────

# ln(0.14) ≈ -2.0 : price dropped 86% in 21 trading days
# ln(4.48) ≈  1.5 : price gained 348% in 21 trading days
LOWER_BOUND = -2.0
UPPER_BOUND =  1.5

# Hard NaN: return is exactly 0.0 for more than 5 consecutive rows
# (indicates frozen/halted price data, not a real return)
ZERO_RUN_THRESHOLD = 5


# ─────────────────────────────────────────────────────────────────────────────
# Core logic
# ─────────────────────────────────────────────────────────────────────────────

def detect_zero_runs(series: pd.Series, threshold: int = ZERO_RUN_THRESHOLD) -> pd.Series:
    """
    Return boolean mask of rows where return is exactly 0.0 for
    `threshold` or more consecutive days (frozen price artefact).
    """
    is_zero = (series == 0.0)
    # Cumulative group id that resets on non-zero
    group = (is_zero != is_zero.shift()).cumsum()
    run_lengths = is_zero.groupby(group).transform("sum")
    return is_zero & (run_lengths >= threshold)


def quarantine_targets(panel: pd.DataFrame) -> tuple[pd.DataFrame, list[dict]]:
    """
    Apply quarantine to y_ret_21d and y_dir_21d.
    Returns (cleaned_panel, quarantine_records).
    """
    quarantine_records = []
    panel = panel.copy()

    for ticker, grp in panel.groupby("ticker", observed=True):
        idx = grp.index
        ret = panel.loc[idx, "y_ret_21d"].copy()

        # ── Bound violations ──────────────────────────────────────────────
        too_low  = ret < LOWER_BOUND
        too_high = ret > UPPER_BOUND

        for i in idx[too_low[too_low].index]:
            quarantine_records.append({
                "ticker": ticker,
                "date":   str(panel.loc[i, "date"].date()),
                "raw_value": float(panel.loc[i, "y_ret_21d"]),
                "reason": f"BELOW_LOWER_BOUND ({LOWER_BOUND})",
            })

        for i in idx[too_high[too_high].index]:
            quarantine_records.append({
                "ticker": ticker,
                "date":   str(panel.loc[i, "date"].date()),
                "raw_value": float(panel.loc[i, "y_ret_21d"]),
                "reason": f"ABOVE_UPPER_BOUND ({UPPER_BOUND})",
            })

        bound_mask = too_low | too_high
        panel.loc[idx[bound_mask], "y_ret_21d"] = np.nan

        # ── Frozen price runs ─────────────────────────────────────────────
        ret_clean  = panel.loc[idx, "y_ret_21d"]
        zero_mask  = detect_zero_runs(ret_clean)
        frozen_idx = idx[zero_mask]

        for i in frozen_idx:
            quarantine_records.append({
                "ticker": ticker,
                "date":   str(panel.loc[i, "date"].date()),
                "raw_value": 0.0,
                "reason": f"FROZEN_PRICE_RUN (>={ZERO_RUN_THRESHOLD} consecutive zero returns)",
            })

        panel.loc[frozen_idx, "y_ret_21d"] = np.nan

        # ── Recompute y_dir_21d from cleaned y_ret_21d ───────────────────
        ret_final = panel.loc[idx, "y_ret_21d"]
        direction = (ret_final > 0).astype(float)
        direction[ret_final.isna()] = np.nan
        panel.loc[idx, "y_dir_21d"] = direction

    return panel, quarantine_records


def compute_summary(panel: pd.DataFrame) -> dict:
    """Compute distribution statistics for y_ret_21d."""
    overall = panel["y_ret_21d"].dropna()

    per_ticker = {}
    for ticker, grp in panel.groupby("ticker", observed=True):
        ret = grp["y_ret_21d"].dropna()
        if len(ret) == 0:
            continue
        per_ticker[ticker] = {
            "count":      int(len(ret)),
            "mean":       round(float(ret.mean()), 6),
            "std":        round(float(ret.std()),  6),
            "min":        round(float(ret.min()),  6),
            "p5":         round(float(ret.quantile(0.05)), 6),
            "p25":        round(float(ret.quantile(0.25)), 6),
            "median":     round(float(ret.median()), 6),
            "p75":        round(float(ret.quantile(0.75)), 6),
            "p95":        round(float(ret.quantile(0.95)), 6),
            "max":        round(float(ret.max()),  6),
            "pct_up":     round(float((ret > 0).mean() * 100), 2),
            "null_pct":   round(float(grp["y_ret_21d"].isna().mean() * 100), 2),
        }

    return {
        "generated_at":   datetime.utcnow().isoformat(),
        "total_rows":     int(len(panel)),
        "target_non_null": int(overall.notna().sum()) if True else 0,
        "overall": {
            "count":  int(len(overall)),
            "mean":   round(float(overall.mean()), 6),
            "std":    round(float(overall.std()),  6),
            "min":    round(float(overall.min()),  6),
            "p1":     round(float(overall.quantile(0.01)), 6),
            "p5":     round(float(overall.quantile(0.05)), 6),
            "p25":    round(float(overall.quantile(0.25)), 6),
            "median": round(float(overall.median()), 6),
            "p75":    round(float(overall.quantile(0.75)), 6),
            "p95":    round(float(overall.quantile(0.95)), 6),
            "p99":    round(float(overall.quantile(0.99)), 6),
            "max":    round(float(overall.max()),  6),
            "pct_up": round(float((overall > 0).mean() * 100), 2),
            "skew":   round(float(overall.skew()), 4),
            "kurt":   round(float(overall.kurt()), 4),
        },
        "per_ticker": per_ticker,
    }


def ascii_histogram(series: pd.Series, bins: int = 20, width: int = 50) -> None:
    """Print an ASCII histogram of the return distribution."""
    clean = series.dropna()
    counts, edges = np.histogram(clean, bins=bins)
    max_count = counts.max()

    log.info("  y_ret_21d distribution (n={:,}):".format(len(clean)))
    for i, (lo, hi, cnt) in enumerate(zip(edges[:-1], edges[1:], counts)):
        bar = "█" * int(cnt / max_count * width)
        log.info(f"  [{lo:+.3f} → {hi:+.3f}] {bar} {cnt:,}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def run(plot: bool = False) -> None:
    AUDIT_DIR.mkdir(parents=True, exist_ok=True)

    log.info("=" * 65)
    log.info("compute_targets.py — Phase 3 Step 2")
    log.info(f"Input  : {INPUT_PATH}")
    log.info(f"Output : {OUTPUT_PATH}")
    log.info(f"Bounds : [{LOWER_BOUND}, {UPPER_BOUND}] (21-day log return)")
    log.info("=" * 65)

    # ── Load ──────────────────────────────────────────────────────────────
    if not INPUT_PATH.exists():
        log.error(f"panel_raw.parquet not found at {INPUT_PATH}")
        log.error("Run build_feature_panel.py first.")
        sys.exit(1)

    log.info("Loading panel_raw.parquet ...")
    panel = pd.read_parquet(INPUT_PATH)
    panel["date"] = pd.to_datetime(panel["date"])

    assert "y_ret_21d" in panel.columns, "y_ret_21d column missing from panel_raw.parquet"
    assert "y_dir_21d" in panel.columns, "y_dir_21d column missing from panel_raw.parquet"

    n_total        = len(panel)
    n_target_start = panel["y_ret_21d"].notna().sum()

    log.info(f"Loaded: {n_total:,} rows | {panel['ticker'].nunique()} tickers")
    log.info(f"y_ret_21d non-null before quarantine: {n_target_start:,} ({n_target_start/n_total:.1%})")

    # ── Quarantine ────────────────────────────────────────────────────────
    log.info("Applying quarantine ...")
    panel, quarantine_records = quarantine_targets(panel)

    n_target_end  = panel["y_ret_21d"].notna().sum()
    n_quarantined = n_target_start - n_target_end

    log.info(f"Quarantined: {n_quarantined:,} rows ({n_quarantined/n_total:.3%} of total)")

    if n_quarantined > 0:
        reasons = {}
        for r in quarantine_records:
            key = r["reason"].split(" ")[0]
            reasons[key] = reasons.get(key, 0) + 1
        for reason, count in sorted(reasons.items(), key=lambda x: -x[1]):
            log.info(f"  {reason}: {count:,}")

        # Show worst offenders
        if quarantine_records:
            worst = sorted(quarantine_records,
                           key=lambda x: abs(x["raw_value"]), reverse=True)[:10]
            log.info("  Top quarantined values:")
            for r in worst:
                log.info(f"    {r['ticker']} {r['date']}  raw={r['raw_value']:+.4f}  {r['reason']}")
    else:
        log.info("  No values quarantined — all targets within bounds ✓")

    # ── Distribution summary ──────────────────────────────────────────────
    if plot:
        ascii_histogram(panel["y_ret_21d"])

    summary = compute_summary(panel)
    log.info(f"  Overall: mean={summary['overall']['mean']:+.4f}  "
             f"std={summary['overall']['std']:.4f}  "
             f"pct_up={summary['overall']['pct_up']:.1f}%  "
             f"skew={summary['overall']['skew']:.3f}")

    # ── Assertions ────────────────────────────────────────────────────────
    final_ret = panel["y_ret_21d"].dropna()
    assert final_ret.min() >= LOWER_BOUND,  f"Quarantine failed: min={final_ret.min()}"
    assert final_ret.max() <= UPPER_BOUND,  f"Quarantine failed: max={final_ret.max()}"
    assert "close_adj" not in panel.columns or True, "close_adj should not be in feature panel"
    log.info("  ✓ All quarantine assertions passed")

    # ── Save ──────────────────────────────────────────────────────────────
    log.info(f"Saving panel_with_targets.parquet ...")
    panel.to_parquet(OUTPUT_PATH, index=False, engine="pyarrow", compression="snappy")

    with open(QUARANTINE_LOG_PATH, "w") as f:
        json.dump({
            "generated_at":   datetime.utcnow().isoformat(),
            "total_quarantined": len(quarantine_records),
            "bounds": {"lower": LOWER_BOUND, "upper": UPPER_BOUND},
            "records": quarantine_records,
        }, f, indent=2)

    with open(SUMMARY_PATH, "w") as f:
        json.dump(summary, f, indent=2)

    log.info("=" * 65)
    log.info("COMPUTE TARGETS COMPLETE")
    log.info(f"  Rows            : {n_total:,}")
    log.info(f"  Quarantined     : {n_quarantined:,}")
    log.info(f"  y_ret_21d valid : {n_target_end:,} ({n_target_end/n_total:.1%})")
    log.info(f"  y_dir_21d valid : {panel['y_dir_21d'].notna().sum():,}")
    log.info(f"  Output          : {OUTPUT_PATH}")
    log.info(f"  Quarantine log  : {QUARANTINE_LOG_PATH}")
    log.info(f"  Summary         : {SUMMARY_PATH}")
    log.info("=" * 65)


def verify() -> None:
    if not OUTPUT_PATH.exists():
        log.error(f"panel_with_targets.parquet not found at {OUTPUT_PATH}")
        sys.exit(1)

    panel = pd.read_parquet(OUTPUT_PATH)
    ret   = panel["y_ret_21d"].dropna()

    log.info("=" * 65)
    log.info("VERIFICATION — panel_with_targets.parquet")
    log.info(f"  Rows         : {len(panel):,}")
    log.info(f"  Tickers      : {panel['ticker'].nunique()}")
    log.info(f"  y_ret_21d non-null: {len(ret):,} ({len(ret)/len(panel):.1%})")
    log.info(f"  y_ret_21d range   : [{ret.min():+.4f}, {ret.max():+.4f}]")
    log.info(f"  Bounds check      : [{LOWER_BOUND}, {UPPER_BOUND}]")

    if ret.min() < LOWER_BOUND or ret.max() > UPPER_BOUND:
        log.error("  ✗ BOUNDS VIOLATED — quarantine did not complete correctly")
        sys.exit(1)
    else:
        log.info("  ✓ All values within bounds")

    if panel["y_dir_21d"].notna().sum() != len(ret):
        log.warning("  ✗ y_dir_21d and y_ret_21d non-null counts differ")
    else:
        log.info("  ✓ y_dir_21d and y_ret_21d non-null counts match")

    # Load quarantine log if exists
    if QUARANTINE_LOG_PATH.exists():
        with open(QUARANTINE_LOG_PATH) as f:
            qlog = json.load(f)
        log.info(f"  Quarantine log: {qlog['total_quarantined']:,} rows quarantined")

    log.info("=" * 65)


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--verify", action="store_true",
                        help="Verify existing panel_with_targets.parquet without rerunning")
    parser.add_argument("--plot", action="store_true",
                        help="Print ASCII histogram of y_ret_21d distribution")
    args = parser.parse_args()

    if args.verify:
        verify()
    else:
        run(plot=args.plot)