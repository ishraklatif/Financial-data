#!/usr/bin/env python3
"""
inspect_panel.py
================
Quick pre-build inspection of the raw parquet files for the 132 usable
ASX 200 tickers. Prints a full summary of:
  - Universe coverage
  - Per-ticker date range and row count
  - Column presence and dtypes
  - Per-column value stats (min, max, mean, null %, sample values)
  - Data quality flags

Usage:
    python inspect_panel.py
    python inspect_panel.py --raw-dir /path/to/data/raw/companies
    python inspect_panel.py --ticker CBA.AX        # single ticker deep-dive
    python inspect_panel.py --cols close_adj volume # specific columns only
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# ── Defaults ─────────────────────────────────────────────────────────────────
DEFAULT_RAW_DIR  = Path.home() / "Documents/financial_data/Financial-data/data/raw/companies"
SUMMARY_JSON     = "_fetch_summary_asx200.json"

# ── Helpers ───────────────────────────────────────────────────────────────────

def load_usable_tickers(raw_dir: Path) -> dict[str, str]:
    """
    Read _fetch_summary_asx200.json and return {ticker: sector}
    for every ticker that passed consistency checks (status != SKIPPED
    and status not in NO_DATA / DOWNLOAD_ERROR).
    """
    summary_path = raw_dir / SUMMARY_JSON
    if not summary_path.exists():
        print(f"ERROR: {summary_path} not found. Run fetch_companies_asx200.py first.")
        sys.exit(1)

    with open(summary_path) as f:
        summary = json.load(f)

    exclude = {"SKIPPED", "NO_DATA", "DOWNLOAD_ERROR", "EMPTY_AFTER_CLEAN", "FAIL"}
    usable  = {
        r["ticker"]: r.get("sector", "Unknown")
        for r in summary["tickers"]
        if r["status"] not in exclude and r.get("rows", 0) > 0
    }
    return usable


def safe_name_to_ticker(stem: str) -> str:
    """Reverse safe_name: CBA_AX → CBA.AX"""
    # Last underscore before two uppercase letters is the dot
    parts = stem.upper().split("_")
    if len(parts) >= 2 and len(parts[-1]) == 2:
        return "_".join(parts[:-1]) + "." + parts[-1]
    return stem.upper()


def load_ticker(parquet_path: Path, ticker: str) -> pd.DataFrame:
    df = pd.read_parquet(parquet_path)
    df["ticker"] = ticker
    df["date"]   = pd.to_datetime(df["date"])
    return df.sort_values("date").reset_index(drop=True)


def col_summary(series: pd.Series) -> dict:
    """Compute per-column stats for a single column."""
    n         = len(series)
    null_n    = series.isna().sum()
    null_pct  = 100 * null_n / max(n, 1)

    if pd.api.types.is_numeric_dtype(series):
        valid = series.dropna()
        return {
            "dtype":    str(series.dtype),
            "null_pct": f"{null_pct:.1f}%",
            "min":      f"{valid.min():.4g}" if len(valid) else "—",
            "max":      f"{valid.max():.4g}" if len(valid) else "—",
            "mean":     f"{valid.mean():.4g}" if len(valid) else "—",
            "std":      f"{valid.std():.4g}"  if len(valid) else "—",
            "zeros":    f"{(valid == 0).sum()} ({100*(valid==0).sum()/max(len(valid),1):.1f}%)",
        }
    else:
        return {
            "dtype":    str(series.dtype),
            "null_pct": f"{null_pct:.1f}%",
            "n_unique": series.nunique(),
            "samples":  list(series.dropna().unique()[:5]),
        }


# ── Per-ticker deep dive ──────────────────────────────────────────────────────

def inspect_ticker(df: pd.DataFrame, ticker: str) -> None:
    print(f"\n{'═'*70}")
    print(f"  {ticker}  |  {len(df):,} rows  |  "
          f"{df['date'].min().date()} → {df['date'].max().date()}")
    print(f"{'═'*70}")

    # Date continuity
    gaps = df["date"].diff().dt.days.dropna()
    max_gap = int(gaps.max()) if len(gaps) else 0
    n_gaps_over_7 = int((gaps > 7).sum())
    print(f"  Max gap : {max_gap} days  |  Gaps > 7d : {n_gaps_over_7}")

    # Column stats
    print(f"\n  {'Column':<22} {'Dtype':<12} {'Nulls':>7}  {'Min':>12}  {'Max':>12}  {'Mean':>12}")
    print(f"  {'─'*22} {'─'*12} {'─'*7}  {'─'*12}  {'─'*12}  {'─'*12}")

    for col in df.columns:
        if col in ("ticker",):
            continue
        s = col_summary(df[col])
        if "min" in s:
            print(f"  {col:<22} {s['dtype']:<12} {s['null_pct']:>7}  "
                  f"{s['min']:>12}  {s['max']:>12}  {s['mean']:>12}")
        else:
            uniq = str(s.get("n_unique", ""))
            samp = str(s.get("samples", ""))[:30]
            print(f"  {col:<22} {s['dtype']:<12} {s['null_pct']:>7}  "
                  f"unique={uniq:<6}  {samp}")


# ── Universe overview ─────────────────────────────────────────────────────────

def inspect_universe(raw_dir: Path, usable: dict[str, str],
                     cols_filter: list[str] | None = None) -> None:

    print(f"\n{'═'*70}")
    print(f"  ASX 200 Universe Inspection")
    print(f"  Usable tickers : {len(usable)}")
    print(f"  Raw dir        : {raw_dir}")
    print(f"{'═'*70}")

    rows_per_ticker = []
    date_mins       = []
    date_maxs       = []
    col_null_counts = {}   # col → list of null %
    col_dtypes      = {}
    sectors         = {}

    found = 0
    missing_files = []

    for ticker, sector in sorted(usable.items()):
        stem = ticker.replace(".", "_")
        path = raw_dir / f"{stem}.parquet"
        if not path.exists():
            missing_files.append(ticker)
            continue

        df = load_ticker(path, ticker)
        found += 1
        rows_per_ticker.append(len(df))
        date_mins.append(df["date"].min())
        date_maxs.append(df["date"].max())
        sectors[ticker] = sector

        # Track column stats
        check_cols = cols_filter if cols_filter else df.columns.tolist()
        for col in check_cols:
            if col not in df.columns:
                continue
            pct = 100 * df[col].isna().sum() / max(len(df), 1)
            col_null_counts.setdefault(col, []).append(pct)
            col_dtypes[col] = str(df[col].dtype)

    # ── Summary table ─────────────────────────────────────────────────────────
    rows_arr = np.array(rows_per_ticker)
    print(f"\n  ROWS PER TICKER")
    print(f"  {'Total rows':>18} : {rows_arr.sum():>10,}")
    print(f"  {'Mean rows/ticker':>18} : {rows_arr.mean():>10,.0f}")
    print(f"  {'Min rows':>18} : {rows_arr.min():>10,}  ({sorted(usable.keys())[np.argmin(rows_arr)]})")
    print(f"  {'Max rows':>18} : {rows_arr.max():>10,}  ({sorted(usable.keys())[np.argmax(rows_arr)]})")
    print(f"  {'Median rows':>18} : {np.median(rows_arr):>10,.0f}")

    print(f"\n  DATE COVERAGE")
    print(f"  {'Earliest date':>18} : {min(date_mins).date()}")
    print(f"  {'Latest date':>18} : {max(date_maxs).date()}")
    tickers_full = sum(1 for d in date_mins if d.year <= 2006)
    print(f"  {'Full history (≤2006)':>18} : {tickers_full} tickers")
    tickers_partial = sum(1 for d in date_mins if 2006 < d.year <= 2015)
    print(f"  {'Partial 2006-2015':>18} : {tickers_partial} tickers")
    tickers_late = sum(1 for d in date_mins if d.year > 2015)
    print(f"  {'Listed after 2015':>18} : {tickers_late} tickers")

    # ── Sector breakdown ──────────────────────────────────────────────────────
    from collections import Counter
    sector_counts = Counter(sectors.values())
    print(f"\n  SECTOR BREAKDOWN")
    for sec, count in sorted(sector_counts.items(), key=lambda x: -x[1]):
        print(f"  {'':4}{sec:<30} : {count:>3} tickers")

    # ── Columns overview ──────────────────────────────────────────────────────
    print(f"\n  COLUMNS  (across {found} tickers)")
    print(f"  {'Column':<22} {'Dtype':<12} {'Null% avg':>10}  {'Null% max':>10}  {'All present?':>13}")
    print(f"  {'─'*22} {'─'*12} {'─'*10}  {'─'*10}  {'─'*13}")

    for col, nulls in sorted(col_null_counts.items()):
        avg_null = np.mean(nulls)
        max_null = np.max(nulls)
        all_present = "YES" if len(nulls) == found else f"NO ({len(nulls)}/{found})"
        dtype = col_dtypes.get(col, "?")
        flag = "  ⚠ " if avg_null > 10 else "    "
        print(f"{flag}{col:<22} {dtype:<12} {avg_null:>9.1f}%  {max_null:>9.1f}%  {all_present:>13}")

    # ── Per-ticker table ──────────────────────────────────────────────────────
    print(f"\n  PER-TICKER SUMMARY")
    print(f"  {'Ticker':<12} {'Sector':<30} {'Rows':>6}  {'From':<12} {'To':<12} {'Status'}")
    print(f"  {'─'*12} {'─'*30} {'─'*6}  {'─'*12} {'─'*12} {'─'*10}")

    for ticker, sector in sorted(usable.items()):
        stem = ticker.replace(".", "_")
        path = raw_dir / f"{stem}.parquet"
        if not path.exists():
            print(f"  {ticker:<12} {sector:<30} {'—':>6}  {'FILE NOT FOUND'}")
            continue
        df = load_ticker(path, ticker)
        flag = " ⚠ partial" if df["date"].min().year > 2010 else ""
        print(f"  {ticker:<12} {sector:<30} {len(df):>6}  "
              f"{str(df['date'].min().date()):<12} {str(df['date'].max().date()):<12}{flag}")

    if missing_files:
        print(f"\n  ⚠  Missing parquet files: {missing_files}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Inspect ASX 200 raw parquets before feature build.")
    parser.add_argument("--raw-dir", default=str(DEFAULT_RAW_DIR),
                        help="Path to data/raw/companies/")
    parser.add_argument("--ticker", default=None,
                        help="Deep-dive a single ticker, e.g. CBA.AX")
    parser.add_argument("--cols", nargs="+", default=None,
                        help="Only show these columns in the universe overview")
    args = parser.parse_args()

    raw_dir = Path(args.raw_dir)
    usable  = load_usable_tickers(raw_dir)
    print(f"Loaded summary: {len(usable)} usable tickers")

    if args.ticker:
        # Single-ticker deep dive
        ticker = args.ticker.upper()
        if ticker not in usable:
            print(f"WARNING: {ticker} not in usable set — showing anyway.")
        stem = ticker.replace(".", "_")
        path = raw_dir / f"{stem}.parquet"
        if not path.exists():
            print(f"ERROR: {path} not found.")
            sys.exit(1)
        df = load_ticker(path, ticker)
        inspect_ticker(df, ticker)
        print(f"\n  Last 5 rows:")
        print(df.tail(5).to_string(index=False))
        print(f"\n  First 5 rows:")
        print(df.head(5).to_string(index=False))
    else:
        inspect_universe(raw_dir, usable, cols_filter=args.cols)

    print()


if __name__ == "__main__":
    main()
