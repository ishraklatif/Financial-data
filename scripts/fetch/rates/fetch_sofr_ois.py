#!/usr/bin/env python3
"""
fetch_sofr_ois.py
=================
Creates a continuous credit stress proxy by splicing the historical
TED spread (TEDRATE, 2005–2022-01-21) with the modern SOFR-OIS spread
(2022-01-24 to present).

Background:
  The TED spread (3-month T-bill yield minus 3-month LIBOR) was the
  standard institutional measure of interbank credit stress. FRED
  discontinued TEDRATE on 2022-01-21 when LIBOR was phased out.

  The modern equivalent is the SOFR-OIS spread:
    SOFR-OIS = SOFR (Secured Overnight Financing Rate)
               minus IORB (Interest on Reserve Balances)

  SOFR is the secured overnight rate. IORB is the Fed's policy rate
  floor. The spread captures the same credit/liquidity premium that
  TED captured, now in the SOFR/SOFR derivatives market.

Splice methodology:
  - Pre-2022-01-21:  Use TEDRATE as-is (already in TEDRATE.parquet)
  - 2022-01-24+:     Fetch SOFR and IORB from FRED, compute SOFR-OIS
  - Overlap check:   Log mean of both series in the 6-month window
                     before 2022-01-21 to document regime shift at splice
  - Continuity:      A 'series_source' column marks which rows are TED
                     vs SOFR-OIS so downstream users can condition on it

Output schema:
  date, value, series_source, series_id, series_name, frequency

  series_source values:
    'TEDRATE'  — original TED spread (pre 2022-01-21)
    'SOFR_OIS' — SOFR minus IORB (post 2022-01-21)

Usage:
    python -m scripts.fetch.rates.fetch_sofr_ois

Verify:
    python -m scripts.fetch.rates.fetch_sofr_ois --verify
    → date_max should be recent, no gap at 2022-01-21

Output:
    data/raw/fred/TED_SPREAD.parquet   (continuous 2005–present)
    data/raw/fred/TED_SPREAD_manifest.json
"""

import argparse
import json
import logging
import os
import ssl
import sys
from datetime import date, datetime
from pathlib import Path

import certifi
import pandas as pd
import yaml
from dotenv import load_dotenv

# SSL fix for Mac
os.environ["SSL_CERT_FILE"]      = certifi.where()
os.environ["REQUESTS_CA_BUNDLE"] = certifi.where()
ssl._create_default_https_context = lambda: ssl.create_default_context(
    cafile=certifi.where()
)

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

load_dotenv(PROJECT_ROOT / ".env")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(PROJECT_ROOT / "logs" / "fetch_sofr_ois.log"),
    ],
)
log = logging.getLogger(__name__)

CONFIG_PATH = PROJECT_ROOT / "config" / "data.yaml"

# TED discontinuation date — last valid TEDRATE observation
TED_END_DATE = "2022-01-21"

# FRED series IDs
SOFR_SERIES  = "SOFR"    # Secured Overnight Financing Rate (daily)
IORB_SERIES  = "IORB"    # Interest on Reserve Balances (daily, replaces IOER)

# Overlap window for splice documentation (days before TED end)
OVERLAP_WINDOW_DAYS = 180


def load_config() -> dict:
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


# ─────────────────────────────────────────────────────────────────────────────
# Fetch from FRED
# ─────────────────────────────────────────────────────────────────────────────

def fetch_fred_series(fred, series_id: str, start: str, end: str) -> pd.Series:
    """Fetch one FRED series. Returns pandas Series indexed by date."""
    log.info(f"Fetching FRED/{series_id} ({start} → {end}) ...")
    raw = fred.get_series(series_id, observation_start=start, observation_end=end)
    log.info(f"  {series_id}: {len(raw)} observations | "
             f"{raw.index.min().date()} → {raw.index.max().date()}")
    return raw


def build_sofr_ois(fred, start: str, end: str) -> pd.DataFrame:
    """
    Fetch SOFR and IORB, compute SOFR-OIS spread.
    Handles the fact that SOFR and IORB may not have identical date coverage.
    """
    sofr = fetch_fred_series(fred, SOFR_SERIES, start, end)
    iorb = fetch_fred_series(fred, IORB_SERIES, start, end)

    # Align on date index — inner join (both must be available)
    df = pd.DataFrame({"SOFR": sofr, "IORB": iorb})
    df.index.name = "date"
    df = df.reset_index()
    df["date"] = pd.to_datetime(df["date"])

    # Drop rows where either series is missing
    n_before = len(df)
    df = df.dropna(subset=["SOFR", "IORB"])
    n_dropped = n_before - len(df)
    if n_dropped > 0:
        log.warning(f"Dropped {n_dropped} rows where SOFR or IORB is missing")

    # Compute spread
    df["value"] = df["SOFR"] - df["IORB"]

    df["series_source"] = "SOFR_OIS"
    df["series_id"]     = "SOFR_OIS"
    df["series_name"]   = "TED_SPREAD"
    df["frequency"]     = "daily"

    log.info(
        f"SOFR-OIS spread: {len(df)} rows | "
        f"{df['date'].min().date()} → {df['date'].max().date()} | "
        f"mean={df['value'].mean():.3f}% | "
        f"min={df['value'].min():.3f}% | max={df['value'].max():.3f}%"
    )

    return df[["date", "value", "series_source", "series_id",
               "series_name", "frequency"]]


def load_ted_historical(fred_dir: Path, global_start: str) -> pd.DataFrame:
    """
    Load existing TEDRATE.parquet. Returns empty DataFrame if not found.
    """
    ted_path = fred_dir / "TEDRATE.parquet"
    if not ted_path.exists():
        log.warning(f"TEDRATE.parquet not found at {ted_path}")
        return pd.DataFrame()

    df = pd.read_parquet(ted_path)
    df["date"] = pd.to_datetime(df["date"])
    df = df[df["date"] >= global_start].copy()

    # Add series_source column if not present
    df["series_source"] = "TEDRATE"
    if "series_id" not in df.columns:
        df["series_id"] = "TEDRATE"
    if "series_name" not in df.columns:
        df["series_name"] = "TED_SPREAD"
    if "frequency" not in df.columns:
        df["frequency"] = "daily"

    # Cap at TED end date
    df = df[df["date"] <= TED_END_DATE].copy()

    log.info(
        f"TED historical: {len(df)} rows | "
        f"{df['date'].min().date()} → {df['date'].max().date()} | "
        f"mean={df['value'].mean():.3f}%"
    )
    return df[["date", "value", "series_source", "series_id",
               "series_name", "frequency"]]


def compute_splice_stats(ted_df: pd.DataFrame,
                         sofr_df: pd.DataFrame) -> dict:
    """
    Document the statistical regime shift at the splice point.
    Compares both series in the 6-month window before TED discontinuation.
    """
    window_start = pd.Timestamp(TED_END_DATE) - pd.Timedelta(days=OVERLAP_WINDOW_DAYS)

    ted_window = ted_df[ted_df["date"] >= window_start]["value"].dropna()

    # SOFR-OIS in same window (if available — SOFR started Nov 2018)
    sofr_window = sofr_df[
        (sofr_df["date"] >= window_start) &
        (sofr_df["date"] <= TED_END_DATE)
    ]["value"].dropna()

    stats = {
        "splice_date": TED_END_DATE,
        "overlap_window_days": OVERLAP_WINDOW_DAYS,
        "ted_window": {
            "start": str(window_start.date()),
            "end":   TED_END_DATE,
            "mean":  round(float(ted_window.mean()), 4) if len(ted_window) else None,
            "std":   round(float(ted_window.std()),  4) if len(ted_window) else None,
            "rows":  len(ted_window),
        },
        "sofr_ois_window": {
            "start": str(window_start.date()),
            "end":   TED_END_DATE,
            "mean":  round(float(sofr_window.mean()), 4) if len(sofr_window) else None,
            "std":   round(float(sofr_window.std()),  4) if len(sofr_window) else None,
            "rows":  len(sofr_window),
        },
    }

    if stats["ted_window"]["mean"] and stats["sofr_ois_window"]["mean"]:
        diff = stats["sofr_ois_window"]["mean"] - stats["ted_window"]["mean"]
        stats["mean_diff_at_splice"] = round(diff, 4)
        log.info(
            f"Splice stats — TED mean: {stats['ted_window']['mean']:.4f}% | "
            f"SOFR-OIS mean: {stats['sofr_ois_window']['mean']:.4f}% | "
            f"diff: {diff:+.4f}%"
        )
    else:
        stats["mean_diff_at_splice"] = None
        log.info("Splice overlap stats: insufficient SOFR-OIS data before TED end")

    return stats


def verify(fred_dir: Path) -> None:
    """Quick verification."""
    path = fred_dir / "TED_SPREAD.parquet"
    if not path.exists():
        log.error("TED_SPREAD.parquet not found")
        return

    df = pd.read_parquet(path)
    df["date"] = pd.to_datetime(df["date"])

    log.info("=" * 55)
    log.info("VERIFICATION — TED_SPREAD.parquet")
    log.info(f"  Rows       : {len(df)}")
    log.info(f"  date_min   : {df['date'].min().date()}")
    log.info(f"  date_max   : {df['date'].max().date()}")
    log.info(f"  Missing    : {df['value'].isna().sum()}")

    sources = df["series_source"].value_counts()
    for src, cnt in sources.items():
        log.info(f"  {src:<12}: {cnt} rows")

    # Check no gap at splice
    splice = pd.Timestamp(TED_END_DATE)
    row_before = df[df["date"] <= splice].tail(1)
    row_after  = df[df["date"] >  splice].head(1)

    if len(row_before) and len(row_after):
        gap_days = (row_after["date"].iloc[0] - row_before["date"].iloc[0]).days
        log.info(f"  Gap at splice: {gap_days} calendar day(s) — "
                 f"{'✓ OK' if gap_days <= 5 else '✗ CHECK'}")
    else:
        log.warning("  Could not check splice gap — missing data near 2022-01-21")

    stale_days = (date.today() - df["date"].max().date()).days
    log.info(f"  Stale days : {stale_days} "
             f"({'✓ current' if stale_days <= 10 else '⚠ check'})")
    log.info("=" * 55)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--verify", action="store_true",
                        help="Verify TED_SPREAD.parquet without re-fetching")
    args = parser.parse_args()

    cfg        = load_config()
    start      = cfg["data"]["start_date"]
    end        = cfg["data"]["end_date"] or date.today().isoformat()
    fred_dir   = PROJECT_ROOT / cfg["raw"]["fred"]

    if args.verify:
        verify(fred_dir)
        return

    api_key = os.getenv("FRED_API_KEY")
    if not api_key or api_key == "your_fred_api_key_here":
        log.error(
            "FRED_API_KEY not set. "
            "Get a free key at: https://fred.stlouisfed.org/docs/api/api_key.html\n"
            "Set it in your .env file: FRED_API_KEY=your_key_here"
        )
        sys.exit(1)

    try:
        from fredapi import Fred
        fred = Fred(api_key=api_key)
    except ImportError:
        log.error("fredapi not installed. Run: pip install fredapi")
        sys.exit(1)

    log.info("=" * 60)
    log.info("fetch_sofr_ois.py — continuous TED/SOFR-OIS credit stress proxy")
    log.info(f"Global start : {start}")
    log.info(f"TED end date : {TED_END_DATE}")
    log.info(f"SOFR-OIS     : {TED_END_DATE} → {end}")
    log.info("=" * 60)

    # ── 1. Load TED historical ────────────────────────────────────────────────
    log.info("Loading TED historical series ...")
    ted_df = load_ted_historical(fred_dir, start)

    if ted_df.empty:
        log.warning(
            "TEDRATE.parquet not found. "
            "Run fetch_fred.py first to fetch TEDRATE, then re-run this script."
        )
        # Proceed anyway — output will be SOFR-OIS only
        log.info("Continuing with SOFR-OIS only (no TED historical)")

    # ── 2. Fetch SOFR-OIS from FRED ───────────────────────────────────────────
    # SOFR data starts 2018-04-02 from FRED. Fetch from TED end onwards.
    sofr_start = "2018-01-01"  # Fetch wider window for splice stats
    log.info("Fetching SOFR and IORB from FRED ...")
    try:
        sofr_df = build_sofr_ois(fred, sofr_start, end)
    except Exception as exc:
        log.error(f"Failed to fetch SOFR-OIS: {exc}")
        sys.exit(1)

    # ── 3. Compute splice statistics ──────────────────────────────────────────
    splice_stats = compute_splice_stats(ted_df, sofr_df)

    # ── 4. Trim SOFR-OIS to post-TED period only (for final output) ──────────
    sofr_post = sofr_df[sofr_df["date"] > TED_END_DATE].copy()
    log.info(
        f"SOFR-OIS post-TED: {len(sofr_post)} rows | "
        f"{sofr_post['date'].min().date() if len(sofr_post) else 'N/A'} → "
        f"{sofr_post['date'].max().date() if len(sofr_post) else 'N/A'}"
    )

    # ── 5. Splice ─────────────────────────────────────────────────────────────
    parts = [p for p in [ted_df, sofr_post] if len(p) > 0]
    if not parts:
        log.error("No data to write — both TED and SOFR-OIS are empty")
        sys.exit(1)

    df_combined = (
        pd.concat(parts, ignore_index=True)
        .drop_duplicates("date", keep="last")
        .sort_values("date")
        .reset_index(drop=True)
    )

    # ── 6. Validate ───────────────────────────────────────────────────────────
    issues = []
    n_missing = int(df_combined["value"].isna().sum())
    if n_missing > 0:
        issues.append({"code": "MISSING_VALUES", "count": n_missing})

    # Check for gap at splice
    splice_ts   = pd.Timestamp(TED_END_DATE)
    rows_before = df_combined[df_combined["date"] <= splice_ts]
    rows_after  = df_combined[df_combined["date"] >  splice_ts]

    if len(rows_before) and len(rows_after):
        gap = (rows_after["date"].iloc[0] - rows_before["date"].iloc[-1]).days
        if gap > 5:
            issues.append({
                "code":   "SPLICE_GAP",
                "detail": f"Gap of {gap} days at splice point {TED_END_DATE}",
            })
            log.warning(f"Gap at splice: {gap} days")
    elif not len(rows_after):
        issues.append({
            "code":   "NO_SOFR_DATA",
            "detail": "No SOFR-OIS data after TED end date",
        })

    status = "WARN" if issues else "OK"

    # ── 7. Save ───────────────────────────────────────────────────────────────
    fred_dir.mkdir(parents=True, exist_ok=True)
    out_path = fred_dir / "TED_SPREAD.parquet"
    df_combined.to_parquet(out_path, index=False)
    log.info(f"Saved TED_SPREAD.parquet → {out_path}")

    manifest = {
        "series_id":          "TED_SPREAD",
        "series_name":        "TED_SPREAD",
        "description":        (
            "Continuous credit stress proxy: TEDRATE (2005–2022-01-21) "
            "spliced with SOFR-OIS spread (2022-01-24 to present). "
            "series_source column indicates which rows are TED vs SOFR-OIS."
        ),
        "output":             str(out_path),
        "status":             status,
        "fetched_at":         datetime.utcnow().isoformat(),
        "ted_end_date":       TED_END_DATE,
        "sofr_ois_start":     str(sofr_post["date"].min().date()) if len(sofr_post) else None,
        "rows":               len(df_combined),
        "rows_ted":           int((df_combined["series_source"] == "TEDRATE").sum()),
        "rows_sofr_ois":      int((df_combined["series_source"] == "SOFR_OIS").sum()),
        "date_min":           str(df_combined["date"].min().date()),
        "date_max":           str(df_combined["date"].max().date()),
        "missing_values":     n_missing,
        "splice_stats":       splice_stats,
        "issues":             issues,
    }

    manifest_path = fred_dir / "TED_SPREAD_manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    log.info("=" * 60)
    log.info("FETCH COMPLETE")
    log.info(f"  Status       : {status}")
    log.info(f"  Rows total   : {len(df_combined)}")
    log.info(f"  Rows TED     : {manifest['rows_ted']}")
    log.info(f"  Rows SOFR-OIS: {manifest['rows_sofr_ois']}")
    log.info(f"  date_min     : {manifest['date_min']}")
    log.info(f"  date_max     : {manifest['date_max']}")
    log.info(f"  Missing      : {n_missing}")
    log.info(f"  Output       → {out_path}")
    log.info(f"  Manifest     → {manifest_path}")
    log.info("=" * 60)
    log.info(
        "NOTE: Update features.yaml — replace xa_ted_spread with xa_ted_spread_continuous. "
        "The series_source column can be used as a regime feature if desired."
    )

    if issues:
        log.warning(f"Issues: {issues}")

    log.info("SOFR-OIS fetch complete. TED gap is closed.")


if __name__ == "__main__":
    main()
