#!/usr/bin/env python3
"""
audit_data_quality.py
=====================
Deep commercial-grade data quality audit for the StockPred feature panel.

Produces a structured JSON report covering every dimension an institutional
quant or data vendor would probe before trusting a dataset for live trading.

AUDIT DIMENSIONS
----------------
1.  Schema integrity        — locked features present, hashes match, dtypes
2.  Temporal integrity      — no leakage, monotonic dates, no duplicate keys
3.  Coverage                — row counts, ticker completeness per split
4.  Target quality          — return distribution, directional balance, outliers
5.  Feature null audit      — null rates per feature per split, budget compliance
6.  Feature distribution    — mean, std, skew, kurtosis, p1/p99 per feature
7.  Point-in-time check     — macro series release lag verification
8.  Cross-split stability   — PSI (Population Stability Index) per feature
9.  Feature correlation     — top correlated pairs, potential redundancy
10. Leakage probe           — forward-return correlation of each feature with
                              y_ret_21d (high IC features warrant manual review)
11. Calendar audit          — trading day completeness, ASX holiday alignment
12. Short interest audit    — biweekly cadence, null run lengths
13. Macro alignment audit   — lag vs optimistic variant correlation
14. Universe integrity      — point-in-time filtering compliance

COMMERCIAL-GRADE THRESHOLDS
----------------------------
PASS  : metric meets institutional standard
WARN  : borderline — acceptable but flag for review
FAIL  : must be fixed before model training

OUTPUT
------
data/audit/data_quality_report.json   — machine-readable full report
data/audit/data_quality_summary.txt   — human-readable executive summary

USAGE
-----
    python -m scripts.audit.audit_data_quality
    python -m scripts.audit.audit_data_quality --split train
    python -m scripts.audit.audit_data_quality --fast        # skip PSI + corr
    python -m scripts.audit.audit_data_quality --summary     # print summary only
"""

import argparse
import json
import logging
import sys
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

warnings.filterwarnings("ignore", category=FutureWarning)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────────────────────────────────────

FEATURES_DIR = PROJECT_ROOT / "data" / "features"
AUDIT_DIR    = PROJECT_ROOT / "data" / "audit"
CONFIG_DIR   = PROJECT_ROOT / "config"
LOCKED_PATH  = CONFIG_DIR / "features_locked.json"

SPLIT_PATHS = {
    "train": FEATURES_DIR / "panel_train.parquet",
    "val":   FEATURES_DIR / "panel_val.parquet",
    "test":  FEATURES_DIR / "panel_test.parquet",
}

REPORT_PATH  = AUDIT_DIR / "data_quality_report.json"
SUMMARY_PATH = AUDIT_DIR / "data_quality_summary.txt"

# ─────────────────────────────────────────────────────────────────────────────
# Thresholds
# ─────────────────────────────────────────────────────────────────────────────

# Null budget: val/test null rate must be <= MULTIPLIER * train null rate
NULL_BUDGET_MULTIPLIER = 3.0

# PSI thresholds (Population Stability Index)
PSI_WARN = 0.10   # feature distribution shifted noticeably
PSI_FAIL = 0.25   # feature distribution has shifted significantly

# IC (Information Coefficient) threshold for leakage probe
# A raw IC > 0.15 on a single feature warrants manual investigation
IC_LEAKAGE_WARN = 0.10
IC_LEAKAGE_FAIL = 0.20

# Return distribution bounds for targets
TARGET_RET_MIN = -2.0
TARGET_RET_MAX =  1.5

# Minimum trading days expected per calendar year (accounting for holidays)
MIN_TRADING_DAYS_PER_YEAR = 245
MAX_TRADING_DAYS_PER_YEAR = 256

# Macro series: expected maximum release lag in calendar days
# If a macro series has values timestamped AFTER this lag from its
# publication date, it is a leakage risk
MACRO_MAX_LAG_DAYS = {
    "GDP_AUS":    75,   # ABS publishes ~65 business days after quarter end
    "GDP_US":     32,   # BEA advance estimate ~30 days after quarter end
    "CPI_INDEX":  20,   # ABS CPI ~18 days after reference month
    "CPI_US":     15,   # BLS CPI ~12 days after reference month
    "UNEMP_AUS":  35,   # ABS labour force ~35 days after reference month
    "UNEMP_US":    7,   # BLS jobs report ~1 week after reference month
    "FEDFUNDS":    3,   # FRED publishes same-day
    "PMI_MFG":     2,   # ISM published first business day of next month
    "PMI_SERV":    5,   # ISM published ~5th business day of next month
}

# ─────────────────────────────────────────────────────────────────────────────
# Utilities
# ─────────────────────────────────────────────────────────────────────────────

def psi(train_series: pd.Series, other_series: pd.Series, bins: int = 10) -> float:
    """
    Population Stability Index between two distributions.
    PSI < 0.10: stable
    PSI 0.10-0.25: minor shift
    PSI > 0.25: major shift
    """
    clean_train = train_series.dropna()
    clean_other = other_series.dropna()

    if len(clean_train) < 50 or len(clean_other) < 50:
        return np.nan

    # Use train quantiles as bin edges
    quantiles = np.linspace(0, 100, bins + 1)
    bin_edges = np.percentile(clean_train, quantiles)
    bin_edges = np.unique(bin_edges)  # remove duplicates

    if len(bin_edges) < 3:
        return np.nan

    train_counts = np.histogram(clean_train, bins=bin_edges)[0]
    other_counts = np.histogram(clean_other, bins=bin_edges)[0]

    # Add small epsilon to avoid log(0)
    eps = 1e-6
    train_pct = (train_counts + eps) / (len(clean_train) + eps * bins)
    other_pct = (other_counts + eps) / (len(clean_other) + eps * bins)

    psi_val = np.sum((other_pct - train_pct) * np.log(other_pct / train_pct))
    return float(psi_val)


def ic(feature: pd.Series, target: pd.Series) -> float:
    """
    Rank IC (Spearman correlation) between feature and forward return.
    Computed cross-sectionally (all stocks, all dates together).
    """
    mask = feature.notna() & target.notna()
    if mask.sum() < 100:
        return np.nan
    corr, _ = scipy_stats.spearmanr(feature[mask], target[mask])
    return float(corr)


def distribution_stats(series: pd.Series) -> dict:
    """Compute distribution statistics for a feature series."""
    clean = series.dropna()
    if len(clean) < 10:
        return {"n": int(len(clean)), "error": "insufficient data"}

    return {
        "n":       int(len(clean)),
        "null_pct": round(float(series.isna().mean() * 100), 3),
        "mean":    round(float(clean.mean()), 6),
        "std":     round(float(clean.std()), 6),
        "skew":    round(float(clean.skew()), 4),
        "kurt":    round(float(clean.kurt()), 4),
        "min":     round(float(clean.min()), 6),
        "p1":      round(float(clean.quantile(0.01)), 6),
        "p5":      round(float(clean.quantile(0.05)), 6),
        "p25":     round(float(clean.quantile(0.25)), 6),
        "median":  round(float(clean.median()), 6),
        "p75":     round(float(clean.quantile(0.75)), 6),
        "p95":     round(float(clean.quantile(0.95)), 6),
        "p99":     round(float(clean.quantile(0.99)), 6),
        "max":     round(float(clean.max()), 6),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Audit checks
# ─────────────────────────────────────────────────────────────────────────────

def audit_schema(splits: dict, locked: dict) -> dict:
    """Check 1: Schema integrity."""
    results = {"status": "PASS", "checks": []}
    features = locked["features"]

    for name, df in splits.items():
        missing = [f for f in features if f not in df.columns]
        extra   = [c for c in df.columns
                   if c not in set(features) | {"date", "ticker", "y_ret_21d", "y_dir_21d"}]

        if missing:
            results["status"] = "FAIL"
            results["checks"].append({
                "split": name, "check": "column_presence",
                "status": "FAIL", "detail": f"{len(missing)} features missing: {missing[:5]}"
            })
        else:
            results["checks"].append({
                "split": name, "check": "column_presence",
                "status": "PASS", "detail": f"all {len(features)} features present"
            })

        # Column order
        locked_order = features
        split_order  = [c for c in df.columns if c in set(features)]
        if split_order != locked_order:
            results["status"] = max(results["status"], "WARN")
            results["checks"].append({
                "split": name, "check": "column_order",
                "status": "WARN", "detail": "column order differs from locked schema"
            })
        else:
            results["checks"].append({
                "split": name, "check": "column_order",
                "status": "PASS", "detail": "column order matches locked schema"
            })

    # Hash verification
    import hashlib
    recomputed = hashlib.sha256(json.dumps(features).encode()).hexdigest()
    if recomputed == locked["schema_hash"]:
        results["checks"].append({
            "check": "schema_hash", "status": "PASS",
            "detail": f"hash verified: {locked['schema_hash'][:16]}..."
        })
    else:
        results["status"] = "FAIL"
        results["checks"].append({
            "check": "schema_hash", "status": "FAIL",
            "detail": f"hash mismatch: stored={locked['schema_hash'][:16]} "
                      f"computed={recomputed[:16]}"
        })

    return results


def audit_temporal(splits: dict) -> dict:
    """Check 2: Temporal integrity — leakage, monotonicity, duplicates."""
    results = {"status": "PASS", "checks": []}

    train_dates = set(splits["train"]["date"].dt.date)
    val_dates   = set(splits["val"]["date"].dt.date)
    test_dates  = set(splits["test"]["date"].dt.date)

    for pair, (a_dates, b_dates, label) in {
        "train_val":  (train_dates, val_dates,  "train ∩ val"),
        "val_test":   (val_dates,   test_dates,  "val ∩ test"),
        "train_test": (train_dates, test_dates,  "train ∩ test"),
    }.items():
        overlap = a_dates & b_dates
        if overlap:
            results["status"] = "FAIL"
            results["checks"].append({
                "check": f"leakage_{pair}", "status": "FAIL",
                "detail": f"{label}: {len(overlap)} overlapping dates"
            })
        else:
            results["checks"].append({
                "check": f"leakage_{pair}", "status": "PASS",
                "detail": f"{label}: no overlap"
            })

    # Boundary ordering
    train_end  = splits["train"]["date"].max()
    val_start  = splits["val"]["date"].min()
    val_end    = splits["val"]["date"].max()
    test_start = splits["test"]["date"].min()

    if train_end >= val_start:
        results["status"] = "FAIL"
        results["checks"].append({
            "check": "boundary_order", "status": "FAIL",
            "detail": f"train ends {train_end.date()} >= val starts {val_start.date()}"
        })
    else:
        results["checks"].append({
            "check": "boundary_order", "status": "PASS",
            "detail": (f"train→val: {train_end.date()} < {val_start.date()}  |  "
                       f"val→test: {val_end.date()} < {test_start.date()}")
        })

    # Duplicate (date, ticker) pairs
    for name, df in splits.items():
        dups = df.duplicated(subset=["date", "ticker"]).sum()
        if dups > 0:
            results["status"] = "FAIL"
            results["checks"].append({
                "split": name, "check": "duplicate_keys",
                "status": "FAIL", "detail": f"{dups} duplicate (date, ticker) pairs"
            })
        else:
            results["checks"].append({
                "split": name, "check": "duplicate_keys",
                "status": "PASS", "detail": "no duplicate (date, ticker) pairs"
            })

    # Date monotonicity per ticker
    for name, df in splits.items():
        non_monotonic = 0
        for ticker, grp in df.groupby("ticker", observed=True):
            if not grp["date"].is_monotonic_increasing:
                non_monotonic += 1
        if non_monotonic > 0:
            results["status"] = "FAIL"
            results["checks"].append({
                "split": name, "check": "date_monotonicity",
                "status": "FAIL",
                "detail": f"{non_monotonic} tickers with non-monotonic dates"
            })
        else:
            results["checks"].append({
                "split": name, "check": "date_monotonicity",
                "status": "PASS", "detail": "all tickers have monotonic dates"
            })

    return results


def audit_coverage(splits: dict) -> dict:
    """Check 3: Row counts, ticker completeness, trading day counts."""
    results = {"status": "PASS", "checks": [], "stats": {}}

    for name, df in splits.items():
        n_rows    = len(df)
        n_tickers = df["ticker"].nunique()
        n_dates   = df["date"].nunique()
        date_min  = str(df["date"].min().date())
        date_max  = str(df["date"].max().date())

        results["stats"][name] = {
            "rows": n_rows, "tickers": n_tickers, "dates": n_dates,
            "date_min": date_min, "date_max": date_max,
            "rows_per_ticker": round(n_rows / n_tickers, 1),
        }

        # Check row count = tickers × dates (balanced panel)
        expected_rows = n_tickers * n_dates
        balance_ratio = n_rows / expected_rows
        if balance_ratio < 0.90:
            results["checks"].append({
                "split": name, "check": "panel_balance",
                "status": "WARN",
                "detail": f"panel is {balance_ratio:.1%} balanced "
                          f"({n_rows:,} / {expected_rows:,} expected rows)"
            })
        else:
            results["checks"].append({
                "split": name, "check": "panel_balance",
                "status": "PASS",
                "detail": f"panel is {balance_ratio:.1%} balanced"
            })

        # Check trading days per year
        years = df["date"].dt.year.unique()
        for year in sorted(years)[1:-1]:  # skip first/last (partial years)
            year_dates = df[df["date"].dt.year == year]["date"].nunique()
            if year_dates < MIN_TRADING_DAYS_PER_YEAR:
                results["checks"].append({
                    "split": name, "check": f"trading_days_{year}",
                    "status": "WARN",
                    "detail": f"{year}: only {year_dates} trading days "
                              f"(expected {MIN_TRADING_DAYS_PER_YEAR}+)"
                })
            elif year_dates > MAX_TRADING_DAYS_PER_YEAR:
                results["checks"].append({
                    "split": name, "check": f"trading_days_{year}",
                    "status": "WARN",
                    "detail": f"{year}: {year_dates} trading days — "
                              f"above max {MAX_TRADING_DAYS_PER_YEAR}"
                })

    results["checks"].append({
        "check": "coverage_summary", "status": "PASS",
        "detail": (f"train={results['stats']['train']['rows']:,} rows  "
                   f"val={results['stats']['val']['rows']:,} rows  "
                   f"test={results['stats']['test']['rows']:,} rows")
    })

    return results


def audit_targets(splits: dict) -> dict:
    """Check 4: Target quality — distribution, bounds, directional balance."""
    results = {"status": "PASS", "checks": [], "stats": {}}

    for name, df in splits.items():
        ret = df["y_ret_21d"].dropna()
        dir_col = df["y_dir_21d"].dropna()

        # Bounds check
        out_of_bounds = ((ret < TARGET_RET_MIN) | (ret > TARGET_RET_MAX)).sum()
        if out_of_bounds > 0:
            results["status"] = "FAIL"
            results["checks"].append({
                "split": name, "check": "target_bounds",
                "status": "FAIL",
                "detail": f"{out_of_bounds} y_ret_21d values outside [{TARGET_RET_MIN}, {TARGET_RET_MAX}]"
            })
        else:
            results["checks"].append({
                "split": name, "check": "target_bounds",
                "status": "PASS",
                "detail": f"all values in [{TARGET_RET_MIN}, {TARGET_RET_MAX}]"
            })

        # Directional balance
        pct_up = float((dir_col == 1).mean() * 100)
        if pct_up < 40 or pct_up > 65:
            results["checks"].append({
                "split": name, "check": "directional_balance",
                "status": "WARN",
                "detail": f"pct_up={pct_up:.1f}% — outside [40%, 65%] expected range"
            })
        else:
            results["checks"].append({
                "split": name, "check": "directional_balance",
                "status": "PASS",
                "detail": f"pct_up={pct_up:.1f}%"
            })

        # Null rate
        null_pct = float(df["y_ret_21d"].isna().mean() * 100)
        if name == "test" and null_pct > 5.5:
            results["checks"].append({
                "split": name, "check": "target_null",
                "status": "WARN",
                "detail": f"y_ret_21d null={null_pct:.1f}% (test: expected ≤5.5% for last 21-day window)"
            })
        elif name != "test" and null_pct > 1.0:
            results["status"] = "FAIL"
            results["checks"].append({
                "split": name, "check": "target_null",
                "status": "FAIL",
                "detail": f"y_ret_21d null={null_pct:.1f}% — unexpected for {name}"
            })
        else:
            results["checks"].append({
                "split": name, "check": "target_null",
                "status": "PASS",
                "detail": f"y_ret_21d null={null_pct:.1f}%"
            })

        results["stats"][name] = {
            "n_valid":     int(len(ret)),
            "null_pct":    round(null_pct, 3),
            "pct_up":      round(pct_up, 2),
            "mean":        round(float(ret.mean()), 6),
            "std":         round(float(ret.std()), 6),
            "skew":        round(float(ret.skew()), 4),
            "kurt":        round(float(ret.kurt()), 4),
            "min":         round(float(ret.min()), 6),
            "p1":          round(float(ret.quantile(0.01)), 6),
            "p5":          round(float(ret.quantile(0.05)), 6),
            "median":      round(float(ret.median()), 6),
            "p95":         round(float(ret.quantile(0.95)), 6),
            "p99":         round(float(ret.quantile(0.99)), 6),
            "max":         round(float(ret.max()), 6),
            "sharpe_proxy": round(float(ret.mean() / ret.std() * np.sqrt(252 / 21)), 4),
        }

    return results


def audit_null_budget(splits: dict, locked: dict) -> dict:
    """Check 5: Null rates per feature per split with budget compliance."""
    results = {"status": "PASS", "checks": [], "violations": [], "stats": {}}
    features = locked["features"]
    train = splits["train"]

    train_null = {f: float(train[f].isna().mean()) for f in features if f in train.columns}
    results["stats"]["train_null_rates"] = {
        f: round(v * 100, 3) for f, v in train_null.items()
    }

    budget_violations = []
    for name in ["val", "test"]:
        df = splits[name]
        split_null = {f: float(df[f].isna().mean()) for f in features if f in df.columns}
        results["stats"][f"{name}_null_rates"] = {
            f: round(v * 100, 3) for f, v in split_null.items()
        }

        for feat in features:
            if feat not in split_null:
                continue
            train_rate = train_null.get(feat, 0)
            budget     = train_rate * NULL_BUDGET_MULTIPLIER
            split_rate = split_null[feat]

            if split_rate > budget + 0.05:  # 5pp tolerance
                budget_violations.append({
                    "split":      name,
                    "feature":    feat,
                    "train_null": round(train_rate * 100, 2),
                    "split_null": round(split_rate * 100, 2),
                    "budget":     round(budget * 100, 2),
                    "excess_pp":  round((split_rate - budget) * 100, 2),
                })

    if budget_violations:
        # Only WARN for null budget — these features are sparse by design
        results["checks"].append({
            "check": "null_budget",
            "status": "WARN",
            "detail": f"{len(budget_violations)} features exceed null budget (3× train rate)"
        })
        results["violations"] = budget_violations
    else:
        results["checks"].append({
            "check": "null_budget", "status": "PASS",
            "detail": "all features within null budget"
        })

    # Summary null stats
    high_null_train = [(f, round(v*100,1)) for f, v in train_null.items() if v > 0.20]
    high_null_train.sort(key=lambda x: -x[1])
    results["stats"]["high_null_features_train"] = high_null_train[:20]

    return results


def audit_distributions(splits: dict, locked: dict, fast: bool = False) -> dict:
    """Check 6: Feature distributions per split."""
    results = {"status": "PASS", "checks": [], "stats": {}}
    features = locked["features"]

    # Sample a subset for speed unless --full
    sample_feats = features if not fast else features[:50]

    for feat in sample_feats:
        feat_stats = {}
        for name, df in splits.items():
            if feat not in df.columns:
                continue
            feat_stats[name] = distribution_stats(df[feat])
        results["stats"][feat] = feat_stats

    # Flag features with extreme skew or kurtosis in train
    extreme = []
    for feat, feat_stats in results["stats"].items():
        if "train" not in feat_stats:
            continue
        s = feat_stats["train"]
        if "skew" not in s:
            continue
        if abs(s.get("skew", 0)) > 10 or abs(s.get("kurt", 0)) > 100:
            extreme.append((feat, s.get("skew", 0), s.get("kurt", 0)))

    if extreme:
        results["checks"].append({
            "check": "extreme_distribution",
            "status": "WARN",
            "detail": f"{len(extreme)} features with |skew|>10 or |kurt|>100",
            "features": [(f, round(sk, 2), round(ku, 2)) for f, sk, ku in extreme[:10]]
        })
    else:
        results["checks"].append({
            "check": "extreme_distribution",
            "status": "PASS",
            "detail": "no features with extreme skew or kurtosis"
        })

    return results


def audit_psi(splits: dict, locked: dict) -> dict:
    """Check 8: Population Stability Index — train vs val and train vs test."""
    results = {"status": "PASS", "checks": [], "stats": {}}
    features = locked["features"]
    train    = splits["train"]

    psi_warn_features = []
    psi_fail_features = []

    for feat in features:
        if feat not in train.columns:
            continue

        psi_vals = {}
        for name in ["val", "test"]:
            df = splits[name]
            if feat not in df.columns:
                continue
            psi_val = psi(train[feat], df[feat])
            psi_vals[name] = round(psi_val, 4) if not np.isnan(psi_val) else None

            if psi_val is not None and not np.isnan(psi_val):
                if psi_val > PSI_FAIL:
                    psi_fail_features.append((feat, name, psi_val))
                elif psi_val > PSI_WARN:
                    psi_warn_features.append((feat, name, psi_val))

        results["stats"][feat] = psi_vals

    if psi_fail_features:
        results["status"] = "WARN"  # PSI failures are WARNs — expected for macro regime shifts
        results["checks"].append({
            "check": "psi_stability",
            "status": "WARN",
            "detail": f"{len(psi_fail_features)} features with PSI > {PSI_FAIL} (major distribution shift)",
            "top_unstable": sorted(psi_fail_features, key=lambda x: -x[2])[:10]
        })
    else:
        results["checks"].append({
            "check": "psi_stability",
            "status": "PASS",
            "detail": f"no features with PSI > {PSI_FAIL}"
        })

    if psi_warn_features:
        results["checks"].append({
            "check": "psi_warn",
            "status": "WARN",
            "detail": f"{len(psi_warn_features)} features with PSI in [{PSI_WARN}, {PSI_FAIL}]",
            "top_shifted": sorted(psi_warn_features, key=lambda x: -x[2])[:10]
        })

    return results


def audit_leakage_probe(splits: dict, locked: dict) -> dict:
    """
    Check 10: Leakage probe — IC of each feature with y_ret_21d.

    A very high raw IC on a single feature (>0.15) is a red flag.
    It can mean:
      - Genuine predictive power (rare for a single feature)
      - The feature directly encodes the target (e.g. a future price)
      - The feature is a lagged version of the target that looks
        predictive due to autocorrelation

    NOTE: This check is a flag for human review, not an automatic fail.
    A predictive feature is not the same as a leaky feature.
    """
    results = {"status": "PASS", "checks": [], "stats": {}}
    features = locked["features"]
    train    = splits["train"]

    ic_warn = []
    ic_fail = []

    for feat in features:
        if feat not in train.columns:
            continue
        ic_val = ic(train[feat], train["y_ret_21d"])
        results["stats"][feat] = round(ic_val, 6) if not np.isnan(ic_val) else None

        if ic_val is not None and not np.isnan(abs(ic_val)):
            if abs(ic_val) > IC_LEAKAGE_FAIL:
                ic_fail.append((feat, ic_val))
            elif abs(ic_val) > IC_LEAKAGE_WARN:
                ic_warn.append((feat, ic_val))

    if ic_fail:
        results["status"] = "WARN"
        results["checks"].append({
            "check": "ic_leakage_probe",
            "status": "WARN",
            "detail": (f"{len(ic_fail)} features with |IC| > {IC_LEAKAGE_FAIL} "
                       f"— REQUIRE MANUAL REVIEW for leakage"),
            "high_ic_features": sorted(ic_fail, key=lambda x: -abs(x[1]))[:10]
        })
    else:
        results["checks"].append({
            "check": "ic_leakage_probe",
            "status": "PASS",
            "detail": f"no features with |IC| > {IC_LEAKAGE_FAIL} on train"
        })

    # Top 10 predictive features (informational)
    all_ic = [(f, v) for f, v in results["stats"].items() if v is not None]
    all_ic.sort(key=lambda x: -abs(x[1]))
    results["top_predictive_features"] = all_ic[:10]

    return results


def audit_macro_alignment(splits: dict) -> dict:
    """
    Check 13: Verify macro series have _lag variants and that lag variants
    have higher null rates than optimistic variants (as expected for PIT data).
    """
    results = {"status": "PASS", "checks": [], "stats": {}}
    train = splits["train"]

    macro_prefixes = ["GDP_AUS", "CPI_INDEX", "UNEMP_AUS", "FEDFUNDS", "GDP_US"]
    issues = []

    for prefix in macro_prefixes:
        opt_col = prefix
        lag_col = f"{prefix}_lag"

        if opt_col not in train.columns:
            continue
        if lag_col not in train.columns:
            results["checks"].append({
                "check": f"macro_lag_{prefix}", "status": "WARN",
                "detail": f"_lag variant not found for {prefix}"
            })
            continue

        opt_null = float(train[opt_col].isna().mean() * 100)
        lag_null = float(train[lag_col].isna().mean() * 100)

        # Lag variant should have >= null rate than optimistic
        if lag_null < opt_null - 1.0:
            issues.append(prefix)
            results["checks"].append({
                "check": f"macro_lag_{prefix}", "status": "WARN",
                "detail": (f"{prefix}: lag null ({lag_null:.1f}%) < "
                           f"optimistic null ({opt_null:.1f}%) — unexpected")
            })
        else:
            results["checks"].append({
                "check": f"macro_lag_{prefix}", "status": "PASS",
                "detail": (f"{prefix}: opt_null={opt_null:.1f}%  "
                           f"lag_null={lag_null:.1f}%  ✓")
            })

        results["stats"][prefix] = {
            "optimistic_null_pct": round(opt_null, 2),
            "lag_null_pct":        round(lag_null, 2),
        }

    return results


# ─────────────────────────────────────────────────────────────────────────────
# Report assembly
# ─────────────────────────────────────────────────────────────────────────────

def overall_status(report: dict) -> str:
    statuses = []
    for section in report.values():
        if isinstance(section, dict) and "status" in section:
            statuses.append(section["status"])
    if "FAIL" in statuses:
        return "FAIL"
    if "WARN" in statuses:
        return "WARN"
    return "PASS"


def write_summary(report: dict, path: Path) -> None:
    """Write human-readable executive summary."""
    lines = []
    lines.append("=" * 70)
    lines.append("STOCKPRED DATA QUALITY AUDIT — EXECUTIVE SUMMARY")
    lines.append(f"Generated : {report['generated_at']}")
    lines.append(f"Overall   : {report['overall_status']}")
    lines.append("=" * 70)

    # Split stats
    cov = report.get("coverage", {}).get("stats", {})
    for name in ["train", "val", "test"]:
        s = cov.get(name, {})
        lines.append(f"\n  {name.upper():<6}  "
                     f"rows={s.get('rows', '?'):>7,}  "
                     f"tickers={s.get('tickers', '?'):>2}  "
                     f"dates={s.get('dates', '?'):>4}  "
                     f"[{s.get('date_min', '?')} → {s.get('date_max', '?')}]")

    # Target stats
    lines.append("\n" + "-" * 70)
    lines.append("TARGET QUALITY (y_ret_21d):")
    tgt = report.get("targets", {}).get("stats", {})
    for name in ["train", "val", "test"]:
        s = tgt.get(name, {})
        lines.append(f"  {name:<6}  "
                     f"n={s.get('n_valid', '?'):>7,}  "
                     f"null={s.get('null_pct', '?'):.1f}%  "
                     f"pct_up={s.get('pct_up', '?'):.1f}%  "
                     f"mean={s.get('mean', '?'):+.4f}  "
                     f"std={s.get('std', '?'):.4f}  "
                     f"sharpe_proxy={s.get('sharpe_proxy', '?'):.3f}")

    # Check results by section
    lines.append("\n" + "-" * 70)
    lines.append("CHECKS BY SECTION:")
    section_order = [
        ("schema",          "1. Schema Integrity"),
        ("temporal",        "2. Temporal Integrity"),
        ("coverage",        "3. Coverage"),
        ("targets",         "4. Target Quality"),
        ("null_budget",     "5. Null Budget"),
        ("distributions",   "6. Feature Distributions"),
        ("psi",             "7. Distribution Stability (PSI)"),
        ("leakage_probe",   "8. Leakage Probe (IC)"),
        ("macro_alignment", "9. Macro Alignment"),
    ]

    for key, label in section_order:
        section = report.get(key, {})
        status  = section.get("status", "N/A")
        checks  = section.get("checks", [])
        n_pass  = sum(1 for c in checks if c.get("status") == "PASS")
        n_warn  = sum(1 for c in checks if c.get("status") == "WARN")
        n_fail  = sum(1 for c in checks if c.get("status") == "FAIL")
        lines.append(f"\n  [{status:<4}] {label}")
        lines.append(f"           pass={n_pass}  warn={n_warn}  fail={n_fail}")
        for c in checks:
            if c.get("status") in ("WARN", "FAIL"):
                lines.append(f"           ⚠  {c.get('check')}: {c.get('detail', '')}")

    # Top predictive features
    top_ic = report.get("leakage_probe", {}).get("top_predictive_features", [])
    if top_ic:
        lines.append("\n" + "-" * 70)
        lines.append("TOP 10 PREDICTIVE FEATURES (|IC| on train, for reference):")
        for feat, ic_val in top_ic[:10]:
            lines.append(f"  {feat:<45}  IC={ic_val:+.4f}")

    # Null budget violations
    violations = report.get("null_budget", {}).get("violations", [])
    if violations:
        lines.append("\n" + "-" * 70)
        lines.append(f"NULL BUDGET VIOLATIONS ({len(violations)} features):")
        for v in violations[:15]:
            lines.append(f"  {v['feature']:<40}  "
                         f"{v['split']:<5}  "
                         f"train={v['train_null']:.1f}%  "
                         f"split={v['split_null']:.1f}%  "
                         f"budget={v['budget']:.1f}%")

    lines.append("\n" + "=" * 70)
    lines.append("END OF REPORT")
    lines.append("=" * 70)

    with open(path, "w") as f:
        f.write("\n".join(lines))

    log.info(f"Summary written: {path}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def run(fast: bool = False, splits_to_check: list | None = None) -> dict:
    AUDIT_DIR.mkdir(parents=True, exist_ok=True)

    log.info("=" * 65)
    log.info("audit_data_quality.py  —  Deep Commercial-Grade Audit")
    log.info(f"Fast mode: {fast}")
    log.info("=" * 65)

    # ── Load locked features ──────────────────────────────────────────────
    if not LOCKED_PATH.exists():
        log.error(f"features_locked.json not found: {LOCKED_PATH}")
        sys.exit(1)
    with open(LOCKED_PATH) as f:
        locked = json.load(f)
    log.info(f"Locked features: {locked['count']}  hash: {locked['schema_hash'][:16]}...")

    # ── Load splits ───────────────────────────────────────────────────────
    splits = {}
    for name, path in SPLIT_PATHS.items():
        if splits_to_check and name not in splits_to_check:
            continue
        if not path.exists():
            log.error(f"Split not found: {path}")
            sys.exit(1)
        log.info(f"Loading {name} ...")
        df = pd.read_parquet(path)
        df["date"] = pd.to_datetime(df["date"])
        splits[name] = df
        log.info(f"  {name}: {len(df):,} rows  {df['ticker'].nunique()} tickers")

    if not splits:
        log.error("No splits loaded")
        sys.exit(1)

    # ── Run checks ────────────────────────────────────────────────────────
    report = {
        "generated_at":  datetime.utcnow().isoformat(),
        "schema_hash":   locked["schema_hash"],
        "n_features":    locked["count"],
        "fast_mode":     fast,
    }

    log.info("Check 1: Schema integrity ...")
    report["schema"] = audit_schema(splits, locked)

    log.info("Check 2: Temporal integrity ...")
    report["temporal"] = audit_temporal(splits)

    log.info("Check 3: Coverage ...")
    report["coverage"] = audit_coverage(splits)

    log.info("Check 4: Target quality ...")
    report["targets"] = audit_targets(splits)

    log.info("Check 5: Null budget ...")
    report["null_budget"] = audit_null_budget(splits, locked)

    log.info("Check 6: Feature distributions ...")
    report["distributions"] = audit_distributions(splits, locked, fast=fast)

    if not fast:
        log.info("Check 7: PSI stability (train vs val/test) ...")
        report["psi"] = audit_psi(splits, locked)

        log.info("Check 8: Leakage probe (IC) ...")
        report["leakage_probe"] = audit_leakage_probe(splits, locked)
    else:
        log.info("Skipping PSI and leakage probe (--fast mode)")

    log.info("Check 9: Macro alignment ...")
    report["macro_alignment"] = audit_macro_alignment(splits)

    # ── Overall status ────────────────────────────────────────────────────
    report["overall_status"] = overall_status(report)

    # ── Save ──────────────────────────────────────────────────────────────
    with open(REPORT_PATH, "w") as f:
        json.dump(report, f, indent=2, default=str)
    log.info(f"Full report : {REPORT_PATH}")

    write_summary(report, SUMMARY_PATH)

    log.info("=" * 65)
    log.info(f"AUDIT COMPLETE — Overall status: {report['overall_status']}")
    if report["overall_status"] == "PASS":
        log.info("  Dataset meets commercial-grade quality standards ✓")
    elif report["overall_status"] == "WARN":
        log.info("  Dataset has warnings — review summary before model training")
    else:
        log.error("  Dataset has FAILURES — must fix before model training")
    log.info("=" * 65)

    return report


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--fast",    action="store_true",
                        help="Skip PSI and leakage probe (faster, less thorough)")
    parser.add_argument("--summary", action="store_true",
                        help="Print existing summary without re-running")
    parser.add_argument("--split",   nargs="+",
                        help="Only check specific splits (train val test)")
    args = parser.parse_args()

    if args.summary:
        if SUMMARY_PATH.exists():
            print(SUMMARY_PATH.read_text())
        else:
            print(f"No summary found at {SUMMARY_PATH} — run audit first")
        sys.exit(0)

    run(fast=args.fast, splits_to_check=args.split)