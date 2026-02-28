#!/usr/bin/env python3
"""
audit_raw_data.py
=================
Validates all raw parquet files across every fetch batch before feature
engineering. Produces a structured JSON report and a human-readable
console summary.

Checks performed per series:
  1. File exists and is readable
  2. Required columns present (date, value or OHLCV)
  3. Date column is datetime, no nulls, no duplicates
  4. Date range vs expected start date
  5. Missing value rate
  6. Impossible values (negative prices, yields > 50%, etc.)
  7. Stale data (last date too far in the past)
  8. Flat/zero variance (stuck sensor)
  9. Row count sanity (too few rows for frequency)

Batches audited:
  Batch 1 — Companies (OHLCV, 51 tickers)
  Batch 2 — Macro indices, volatility, FX, commodities, credit, sector
  Batch 3 — FRED (US macro)
  Batch 4 — RBA rates, RBA macro, RBA credit, ABS gdp, ABS wpi
  Batch 5 — ASIC short interest

Usage:
    python -m scripts.audit.audit_raw_data
    python -m scripts.audit.audit_raw_data --verbose

Output:
    data/audit/raw_audit_{timestamp}.json
    data/audit/raw_audit_latest.json   (always overwritten)
"""

import argparse
import json
import logging
import sys
from datetime import datetime, date, timedelta
from pathlib import Path

import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

CONFIG_PATH = PROJECT_ROOT / "config" / "data.yaml"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)

# How stale is too stale (calendar days since last observation)
STALE_DAILY_DAYS    = 10   # daily series
STALE_MONTHLY_DAYS  = 60   # monthly series
STALE_QUARTERLY_DAYS = 120  # quarterly series

# Minimum row counts by frequency
MIN_ROWS = {
    "daily":     2000,   # ~8 years of trading days
    "monthly":   100,
    "quarterly": 40,
    "weekly":    400,
}

# Value sanity bounds: (min, max) — None = no check on that side
VALUE_BOUNDS = {
    # Prices — must be positive
    "open":   (0, None),
    "high":   (0, None),
    "low":    (0, None),
    "close":  (0, None),
    "adj_close": (0, None),
    "value":  (None, None),  # generic — depends on series
    # Short interest
    "short_pct": (0, 100),
    "short_positions": (0, None),
    "total_issued": (0, None),
}

# Yield/rate series — flag if value column > 50 (clearly wrong)
YIELD_SERIES = {
    "CASH_RATE", "YIELD_2Y", "YIELD_10Y", "CPI_YOY",
    "F3_A_YIELD_3Y", "F3_A_YIELD_5Y", "F3_A_YIELD_7Y", "F3_A_YIELD_10Y",
    "F3_BBB_YIELD_3Y", "F3_BBB_YIELD_5Y", "F3_BBB_YIELD_7Y", "F3_BBB_YIELD_10Y",
}


def load_config() -> dict:
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


# ─────────────────────────────────────────────────────────────────────────────
# Audit a single parquet file
# ─────────────────────────────────────────────────────────────────────────────

def audit_file(
    path: Path,
    canonical_name: str,
    expected_start: date,
    frequency: str,
    series_type: str = "generic",  # "ohlcv" | "single_value" | "short_interest"
    verbose: bool = False,
) -> dict:
    """Audit one parquet file. Returns a result dict."""

    issues   = []
    warnings = []

    # ── 1. File exists ────────────────────────────────────────────────────────
    if not path.exists():
        return {
            "canonical_name": canonical_name,
            "path":           str(path),
            "status":         "ERROR",
            "issues":         [{"code": "FILE_NOT_FOUND", "detail": str(path)}],
            "warnings":       [],
            "rows":           0,
            "date_min":       None,
            "date_max":       None,
            "missing_pct":    None,
            "frequency":      frequency,
        }

    # ── 2. Load ───────────────────────────────────────────────────────────────
    try:
        df = pd.read_parquet(path)
    except Exception as e:
        return {
            "canonical_name": canonical_name,
            "path":           str(path),
            "status":         "ERROR",
            "issues":         [{"code": "UNREADABLE", "detail": str(e)[:200]}],
            "warnings":       [],
            "rows":           0,
            "date_min":       None,
            "date_max":       None,
            "missing_pct":    None,
            "frequency":      frequency,
        }

    rows = len(df)

    # ── 3. Date column ────────────────────────────────────────────────────────
    date_col = None
    for c in ["date", "Date"]:
        if c in df.columns:
            date_col = c
            break

    if date_col is None:
        issues.append({"code": "NO_DATE_COLUMN", "columns": list(df.columns)})
    else:
        df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
        n_null_dates = int(df[date_col].isna().sum())
        if n_null_dates > 0:
            issues.append({"code": "NULL_DATES", "count": n_null_dates})

        df = df.dropna(subset=[date_col])
        n_dupe_dates = int(df.duplicated(subset=[date_col]).sum())

        # For multi-ticker files (companies, short interest) duplicates
        # per date are expected — check per-ticker instead
        if series_type in ("ohlcv", "short_interest"):
            ticker_col = "ticker" if "ticker" in df.columns else "product_code"
            if ticker_col in df.columns:
                n_dupe_dates = int(
                    df.duplicated(subset=[date_col, ticker_col]).sum()
                )

        if n_dupe_dates > 0:
            issues.append({"code": "DUPLICATE_DATES", "count": n_dupe_dates})

        date_min = df[date_col].min().date()
        date_max = df[date_col].max().date()

        # ── 4. Date range ─────────────────────────────────────────────────────
        days_late = (date_min - expected_start).days
        if days_late > 365:
            warnings.append({
                "code":   "LATE_START",
                "detail": f"Data starts {date_min}, expected ~{expected_start} "
                          f"({days_late} days late)"
            })

        # ── 5. Stale check ────────────────────────────────────────────────────
        stale_threshold = {
            "daily":     STALE_DAILY_DAYS,
            "monthly":   STALE_MONTHLY_DAYS,
            "quarterly": STALE_QUARTERLY_DAYS,
            "weekly":    STALE_DAILY_DAYS * 3,
        }.get(frequency, STALE_DAILY_DAYS * 2)

        days_stale = (date.today() - date_max).days
        if days_stale > stale_threshold:
            warnings.append({
                "code":   "STALE_DATA",
                "detail": f"Last observation {date_max} is {days_stale} days ago "
                          f"(threshold: {stale_threshold})"
            })

    # ── 6. Row count ──────────────────────────────────────────────────────────
    min_rows = MIN_ROWS.get(frequency, 50)
    if series_type in ("ohlcv", "short_interest"):
        # Multi-ticker: check rows per ticker
        ticker_col = "ticker" if "ticker" in df.columns else (
            "product_code" if "product_code" in df.columns else None
        )
        if ticker_col:
            per_ticker = df.groupby(ticker_col).size()
            thin_tickers = per_ticker[per_ticker < min_rows]
            if len(thin_tickers) > 0:
                warnings.append({
                    "code":    "LOW_ROW_COUNT_PER_TICKER",
                    "tickers": thin_tickers.index.tolist()[:10],
                    "min_rows_found": int(thin_tickers.min()),
                })
    else:
        if rows < min_rows:
            issues.append({
                "code":    "LOW_ROW_COUNT",
                "rows":    rows,
                "minimum": min_rows,
            })

    # ── 7. Missing values ─────────────────────────────────────────────────────
    value_cols = [c for c in df.columns
                  if c not in (date_col, "Date", "series_id", "series_name",
                               "ticker", "product_code", "frequency", "source")]
    total_cells  = rows * len(value_cols) if value_cols else 1
    missing_cells = int(df[value_cols].isna().sum().sum()) if value_cols else 0
    missing_pct  = round(100 * missing_cells / total_cells, 2) if total_cells else 0

    if missing_pct > 20:
        issues.append({"code": "HIGH_MISSING", "pct": missing_pct})
    elif missing_pct > 5:
        warnings.append({"code": "ELEVATED_MISSING", "pct": missing_pct})

    # ── 8. Value bounds ───────────────────────────────────────────────────────
    for col in value_cols:
        lo, hi = VALUE_BOUNDS.get(col, (None, None))
        series_vals = pd.to_numeric(df[col], errors="coerce").dropna()
        if len(series_vals) == 0:
            continue
        if lo is not None and (series_vals < lo).any():
            n_bad = int((series_vals < lo).sum())
            issues.append({
                "code":   "BELOW_MIN",
                "column": col,
                "min":    lo,
                "count":  n_bad,
            })
        if hi is not None and (series_vals > hi).any():
            n_bad = int((series_vals > hi).sum())
            issues.append({
                "code":   "ABOVE_MAX",
                "column": col,
                "max":    hi,
                "count":  n_bad,
            })

    # Yield/rate specific check
    if canonical_name in YIELD_SERIES and "value" in df.columns:
        vals = pd.to_numeric(df["value"], errors="coerce").dropna()
        if len(vals) > 0 and (vals > 50).any():
            issues.append({
                "code":   "IMPLAUSIBLE_YIELD",
                "detail": f"Values > 50% found: max={float(vals.max()):.2f}",
            })

    # ── 9. Zero variance ─────────────────────────────────────────────────────
    if series_type == "single_value" and "value" in df.columns:
        vals = pd.to_numeric(df["value"], errors="coerce").dropna()
        if len(vals) > 10 and vals.std() == 0:
            issues.append({"code": "ZERO_VARIANCE", "column": "value"})

    # ── Final status ──────────────────────────────────────────────────────────
    if issues:
        status = "ERROR"
    elif warnings:
        status = "WARN"
    else:
        status = "OK"

    result = {
        "canonical_name": canonical_name,
        "path":           str(path),
        "status":         status,
        "issues":         issues,
        "warnings":       warnings,
        "rows":           rows,
        "date_min":       str(date_min) if date_col and rows > 0 else None,
        "date_max":       str(date_max) if date_col and rows > 0 else None,
        "missing_pct":    missing_pct,
        "frequency":      frequency,
        "columns":        list(df.columns),
    }

    if verbose:
        symbol = {"OK": "✓", "WARN": "⚠", "ERROR": "✗"}[status]
        log.info(f"  {symbol} {canonical_name:<35} {status:<6} "
                 f"rows={rows:<7} missing={missing_pct:.1f}%  "
                 f"{str(date_min) if date_col else 'N/A'} → "
                 f"{str(date_max) if date_col else 'N/A'}")
        for iss in issues:
            log.info(f"      ERROR: {iss}")
        for w in warnings:
            log.info(f"      WARN:  {w}")

    return result


# ─────────────────────────────────────────────────────────────────────────────
# Build catalogue of all series to audit
# ─────────────────────────────────────────────────────────────────────────────

def build_catalogue(cfg: dict, raw: dict) -> list[dict]:
    """Return list of dicts describing every series to audit."""

    start = date.fromisoformat(cfg["data"]["start_date"])
    entries = []

    def add(name, path, freq, stype="single_value"):
        entries.append({
            "canonical_name":   name,
            "path":             PROJECT_ROOT / path,
            "frequency":        freq,
            "series_type":      stype,
            "expected_start":   start,
        })

    # ── Batch 1: Companies ────────────────────────────────────────────────────
    companies_dir = PROJECT_ROOT / raw["companies"]
    tickers = list(cfg["companies"]["tickers"].keys())
    for ticker in tickers:
        fname = ticker.replace(".AX", "_AX") + ".parquet"
        add(ticker, companies_dir / fname, "daily", "ohlcv")

    # ── Batch 2: Macro indices ────────────────────────────────────────────────
    macro_dir = PROJECT_ROOT / raw["macro_indices"]
    for canonical, yf_ticker in cfg["macro"]["indices"].items():
        fname = canonical + ".parquet"
        add(canonical, macro_dir / fname, "daily", "ohlcv")

    vol_dir = PROJECT_ROOT / raw["macro_vol"]
    for canonical in cfg["macro"]["volatility"].keys():
        add(canonical, vol_dir / (canonical + ".parquet"), "daily", "single_value")

    fx_dir = PROJECT_ROOT / raw["fx"]
    for canonical in cfg["market"]["fx"].keys():
        add(canonical, fx_dir / (canonical + ".parquet"), "daily", "single_value")

    comm_dir = PROJECT_ROOT / raw["commodities"]
    for canonical in cfg["market"]["commodities"].keys():
        add(canonical, comm_dir / (canonical + ".parquet"), "daily", "single_value")

    sector_dir = PROJECT_ROOT / raw["sector"]
    for ticker in cfg["sector"]["tickers"]:
        add(ticker, sector_dir / (ticker + ".parquet"), "daily", "ohlcv")

    # ── Batch 3: FRED ─────────────────────────────────────────────────────────
    fred_dir = PROJECT_ROOT / raw["fred"]
    # Map canonical name → (actual filename on disk, frequency)
    fred_file_map = {
        "CPI":       ("CPIAUCSL.parquet",          "monthly"),
        "PCEPI":     ("PCEPI.parquet",              "monthly"),
        "UNEMP":     ("UNRATE.parquet",             "monthly"),
        "JOLTS":     ("JTSJOL.parquet",             "monthly"),
        "FEDFUNDS":  ("FEDFUNDS.parquet",           "monthly"),
        "DGS2":      ("DGS2.parquet",               "daily"),
        "DGS10":     ("DGS10.parquet",              "daily"),
        "DGS30":     ("DGS30.parquet",              "daily"),
        "DGS3MO":    ("DGS3MO.parquet",             "daily"),
        "GDP":       ("GDPC1.parquet",              "quarterly"),
        "INDPRO":    ("INDPRO.parquet",             "monthly"),
        "PMI_MFG":   ("IPMAN.parquet",              "monthly"),
        "PMI_SERV":  ("SRVPRD.parquet",             "monthly"),
        "HY_SPREAD": ("BAMLH0A0HYM2.parquet",       "daily"),
        "IG_SPREAD": ("BAMLCC0A1AAATRIV.parquet",   "daily"),
        "TED":       ("TEDRATE.parquet",            "daily"),
        "UMICH":     ("UMCSENT.parquet",            "monthly"),
    }
    for canonical in cfg["fred"]["series"].keys():
        fname, freq = fred_file_map.get(canonical, (canonical + ".parquet", "daily"))
        add(canonical, fred_dir / fname, freq)

    # ── Batch 4: RBA rates ────────────────────────────────────────────────────
    rba_rates_dir = PROJECT_ROOT / raw["rba_rates"]
    for name, freq in [("CASH_RATE", "daily"), ("YIELD_2Y", "daily"), ("YIELD_10Y", "daily")]:
        add(name, rba_rates_dir / (name + ".parquet"), freq)

    rba_macro_dir = PROJECT_ROOT / "data" / "raw" / "rba" / "macro"
    for name, freq in [("CPI_INDEX", "quarterly"), ("CPI_YOY", "quarterly"),
                       ("UNEMP", "monthly")]:
        add(name, rba_macro_dir / (name + ".parquet"), freq)

    rba_credit_dir = PROJECT_ROOT / raw["rba_credit"]
    for name in ["F3_A_YIELD_3Y", "F3_A_YIELD_5Y", "F3_A_YIELD_7Y", "F3_A_YIELD_10Y",
                 "F3_BBB_YIELD_3Y", "F3_BBB_YIELD_5Y", "F3_BBB_YIELD_7Y", "F3_BBB_YIELD_10Y"]:
        add(name, rba_credit_dir / (name + ".parquet"), "monthly")

    # ── Batch 4: ABS ──────────────────────────────────────────────────────────
    abs_gdp_dir = PROJECT_ROOT / raw["abs_gdp"]
    for name in ["GDP", "GDP_PCA", "HSR", "TOT"]:
        add(name, abs_gdp_dir / (name + ".parquet"), "quarterly")

    abs_wpi_dir = PROJECT_ROOT / raw["abs_wpi"]
    add("WPI", abs_wpi_dir / "WPI.parquet", "quarterly")

    # ── Batch 5: Short interest ───────────────────────────────────────────────
    short_dir = PROJECT_ROOT / raw["short_interest"]
    add("SHORT_INTEREST", short_dir / "SHORT_INTEREST.parquet", "daily", "short_interest")

    return entries


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    cfg = load_config()
    raw = cfg["raw"]

    audit_dir = PROJECT_ROOT / cfg["output"]["audit"]
    audit_dir.mkdir(parents=True, exist_ok=True)

    log.info("=" * 65)
    log.info("RAW DATA AUDIT")
    log.info(f"Project root : {PROJECT_ROOT}")
    log.info(f"Config       : {CONFIG_PATH}")
    log.info("=" * 65)

    catalogue = build_catalogue(cfg, raw)
    log.info(f"Series to audit: {len(catalogue)}")

    results = []
    batch_labels = {
        "ohlcv":          "Companies / Indices / Sector",
        "single_value":   "Macro / FRED / RBA / ABS",
        "short_interest": "Short Interest",
    }

    current_type = None
    for entry in catalogue:
        if args.verbose and entry["series_type"] != current_type:
            current_type = entry["series_type"]
            log.info(f"\n── {batch_labels.get(current_type, current_type)} ──")

        result = audit_file(
            path           = entry["path"],
            canonical_name = entry["canonical_name"],
            expected_start = entry["expected_start"],
            frequency      = entry["frequency"],
            series_type    = entry["series_type"],
            verbose        = args.verbose,
        )
        results.append(result)

    # ── Summary ───────────────────────────────────────────────────────────────
    n_ok   = sum(1 for r in results if r["status"] == "OK")
    n_warn = sum(1 for r in results if r["status"] == "WARN")
    n_err  = sum(1 for r in results if r["status"] == "ERROR")

    errors   = [r for r in results if r["status"] == "ERROR"]
    warnings = [r for r in results if r["status"] == "WARN"]

    log.info("\n" + "=" * 65)
    log.info("AUDIT SUMMARY")
    log.info(f"  Total   : {len(results)}")
    log.info(f"  OK      : {n_ok}")
    log.info(f"  WARN    : {n_warn}")
    log.info(f"  ERROR   : {n_err}")
    log.info("=" * 65)

    if errors:
        log.info("\nERRORS:")
        for r in errors:
            log.info(f"  ✗ {r['canonical_name']}")
            for iss in r["issues"]:
                log.info(f"      {iss}")

    if warnings:
        log.info("\nWARNINGS:")
        for r in warnings:
            log.info(f"  ⚠ {r['canonical_name']}")
            for w in r["warnings"]:
                log.info(f"      {w}")

    # ── Save report ───────────────────────────────────────────────────────────
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    report = {
        "generated_at":   datetime.utcnow().isoformat(),
        "project_root":   str(PROJECT_ROOT),
        "total":          len(results),
        "ok":             n_ok,
        "warn":           n_warn,
        "error":          n_err,
        "series":         results,
    }

    ts_path     = audit_dir / f"raw_audit_{timestamp}.json"
    latest_path = audit_dir / "raw_audit_latest.json"

    for p in [ts_path, latest_path]:
        with open(p, "w") as f:
            json.dump(report, f, indent=2, default=str)

    log.info(f"\nReport saved → {ts_path}")
    log.info(f"Latest link  → {latest_path}")

    if n_err > 0:
        log.error(f"\n{n_err} ERROR(s) found — fix before proceeding to feature engineering")
        sys.exit(1)

    log.info("\nAudit complete. Safe to proceed to feature engineering.")


if __name__ == "__main__":
    main()