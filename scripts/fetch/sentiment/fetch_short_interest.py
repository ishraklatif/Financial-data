#!/usr/bin/env python3
"""
fetch_short_interest.py
=======================
Downloads ASIC daily aggregated short position reports and extracts
short interest data for the ASX universe defined in data.yaml.

Source:
  https://download.asic.gov.au/short-selling/RR{YYYYMMDD}-001-SSDailyAggShortPos.csv

Published T+4 business days after the position date.
Columns per file: Product, Product Code, Reported Short Positions,
                  Total Product in Issue, % of Total Product in Issue

We extract:
  - short_positions:  raw count of reported short positions
  - total_issued:     total shares on issue
  - short_pct:        short positions as % of total issued (key feature)

ASIC data starts from 2010-06-01 (when mandatory reporting began).
We fetch from max(start_date, 2010-06-01).

Usage:
    python -m scripts.fetch.sentiment.fetch_short_interest

    # Incremental update (only fetch missing dates):
    python -m scripts.fetch.sentiment.fetch_short_interest --incremental

Output:
    data/raw/sentiment/short_interest/SHORT_INTEREST.parquet
    data/raw/sentiment/short_interest/_fetch_summary.json
"""

import argparse
import json
import logging
import sys
import time
from datetime import date, datetime, timedelta
from pathlib import Path

import pandas as pd
import requests
import yaml
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

load_dotenv(PROJECT_ROOT / ".env")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(PROJECT_ROOT / "logs" / "fetch_short_interest.log"),
    ],
)
log = logging.getLogger(__name__)

CONFIG_PATH = PROJECT_ROOT / "config" / "data.yaml"

# ASIC mandatory reporting started 2010-06-01
# Archive confirmed: Jan 2012 = 404, Jun 2012 = 200. Start Feb 2012 to be safe.
ASIC_START = date(2012, 2, 1)

# URL pattern — ASIC publishes T+4 business days after position date
ASIC_URL = "https://download.asic.gov.au/short-selling/RR{date}-001-SSDailyAggShortPos.csv"

# Polite delay between requests
REQUEST_DELAY = 0.5  # seconds

# Max consecutive 404s before assuming we've hit the end of available data
# 30 handles long holiday runs (Christmas+NY ~10 days, Easter ~4) without
# giving up prematurely on the archive boundary.
MAX_CONSECUTIVE_404 = 30


def load_config() -> dict:
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


def get_asx_codes(cfg: dict) -> set[str]:
    """Extract bare ASX codes (without .AX suffix) from config."""
    tickers = cfg["companies"]["tickers"]
    return {t.replace(".AX", "").upper() for t in tickers}


def business_days_between(start: date, end: date) -> list[date]:
    """Return list of weekdays (Mon-Fri) between start and end inclusive."""
    days = []
    current = start
    while current <= end:
        if current.weekday() < 5:  # Monday=0, Friday=4
            days.append(current)
        current += timedelta(days=1)
    return days


def fetch_one_day(report_date: date, session: requests.Session) -> pd.DataFrame | None:
    """
    Download and parse one ASIC short position CSV.
    Returns DataFrame with columns: [product_code, short_positions,
    total_issued, short_pct] or None if file not available.
    """
    url = ASIC_URL.format(date=report_date.strftime("%Y%m%d"))

    try:
        resp = session.get(url, timeout=30)
    except requests.RequestException as exc:
        log.warning(f"[{report_date}] Request error: {exc}")
        return None

    if resp.status_code == 404:
        return None  # Not published for this date (holiday or not yet available)

    if resp.status_code != 200:
        log.warning(f"[{report_date}] HTTP {resp.status_code}")
        return None

    try:
        from io import StringIO
        raw = resp.content
        # Older ASIC files (pre ~2016) are UTF-16 LE with BOM (0xff 0xfe).
        # Newer files are UTF-8. Detect from BOM bytes.
        if raw[:2] == b"\xff\xfe":
            text = raw.decode("utf-16-le").lstrip("\ufeff")
        elif raw[:3] == b"\xef\xbb\xbf":
            text = raw.decode("utf-8-sig")
        else:
            text = raw.decode("utf-8", errors="replace")
        # Old files (pre ~2016) are tab-separated; newer files are comma-separated.
        # Detect by checking if the first line contains tabs.
        first_line = text.split("\n")[0]
        sep = "\t" if "\t" in first_line else ","
        df = pd.read_csv(StringIO(text), sep=sep)
        # Strip whitespace from all column names (TSV files have extra spaces)
        df.columns = [c.strip() for c in df.columns]
    except Exception as exc:
        log.warning(f"[{report_date}] CSV parse error: {exc}")
        return None

    # Normalise column names — ASIC has changed format/delimiter over the years.
    # Order matters: match "%" first (most specific) before "short positions"
    # because the old TSV % column name contains the words "short positions" too.
    col_map = {}
    for c in df.columns:
        cl = c.strip().lower()
        if "product code" in cl or cl == "code":
            col_map[c] = "product_code"
        elif "%" in cl or cl.startswith("percent"):
            col_map[c] = "short_pct"
        elif "reported short" in cl:
            col_map[c] = "short_positions"
        elif "total product" in cl or "total in issue" in cl or "total issued" in cl:
            col_map[c] = "total_issued"
    df = df.rename(columns=col_map)
    # Strip whitespace from product_code values (TSV files have trailing spaces)
    if "product_code" in df.columns:
        df["product_code"] = df["product_code"].astype(str).str.strip()

    required = {"product_code", "short_positions", "total_issued", "short_pct"}
    missing_cols = required - set(df.columns)
    if missing_cols:
        log.warning(f"[{report_date}] Missing columns after rename: {missing_cols}. Got: {list(df.columns)}")
        return None

    df["product_code"]    = df["product_code"].astype(str).str.strip().str.upper()
    df["short_positions"] = pd.to_numeric(df["short_positions"], errors="coerce")
    df["total_issued"]    = pd.to_numeric(df["total_issued"],    errors="coerce")
    df["short_pct"]       = pd.to_numeric(df["short_pct"],       errors="coerce")

    return df[["product_code", "short_positions", "total_issued", "short_pct"]]


def get_existing_dates(out_path: Path) -> set[date]:
    """Return set of dates already in the output parquet."""
    if not out_path.exists():
        return set()
    try:
        df = pd.read_parquet(out_path, columns=["date"])
        return set(df["date"].dt.date.unique())
    except Exception:
        return set()


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--incremental", action="store_true",
        help="Only fetch dates not already in the output parquet"
    )
    parser.add_argument(
        "--start", default=None,
        help="Override start date (YYYY-MM-DD)"
    )
    args = parser.parse_args()

    cfg     = load_config()
    tickers = get_asx_codes(cfg)

    config_start = date.fromisoformat(cfg["data"]["start_date"])
    fetch_start  = max(config_start, ASIC_START)
    fetch_end    = date.today()

    if args.start:
        fetch_start = max(date.fromisoformat(args.start), ASIC_START)

    out_dir  = PROJECT_ROOT / cfg["raw"]["short_interest"]
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "SHORT_INTEREST.parquet"

    log.info("=" * 60)
    log.info("fetch_short_interest.py — ASIC short position reports")
    log.info(f"Universe   : {len(tickers)} ASX tickers")
    log.info(f"Date range : {fetch_start} → {fetch_end}")
    log.info(f"Mode       : {'incremental' if args.incremental else 'full'}")
    log.info("=" * 60)

    # Determine which dates to fetch
    all_days = business_days_between(fetch_start, fetch_end)

    if args.incremental:
        existing = get_existing_dates(out_path)
        days_to_fetch = [d for d in all_days if d not in existing]
        log.info(f"Incremental: {len(existing)} dates already fetched, "
                 f"{len(days_to_fetch)} remaining")
    else:
        days_to_fetch = all_days
        log.info(f"Full fetch: {len(days_to_fetch)} business days to attempt")

    if not days_to_fetch:
        log.info("Nothing to fetch — already up to date")
        return

    # Fetch
    session = requests.Session()
    session.headers["User-Agent"] = "Mozilla/5.0 (research data pipeline)"

    all_records = []
    n_ok        = 0
    n_404       = 0
    n_err       = 0
    consecutive_404 = 0

    for i, report_date in enumerate(days_to_fetch):
        if i % 50 == 0:
            log.info(f"Progress: {i}/{len(days_to_fetch)} | "
                     f"ok={n_ok} 404={n_404} err={n_err}")

        df_day = fetch_one_day(report_date, session)

        if df_day is None:
            n_404 += 1
            consecutive_404 += 1
            if consecutive_404 >= MAX_CONSECUTIVE_404:
                log.info(f"[{report_date}] {MAX_CONSECUTIVE_404} consecutive 404s — "
                         f"assuming end of available data")
                break
            time.sleep(REQUEST_DELAY)
            continue

        consecutive_404 = 0
        n_ok += 1

        # Filter to our universe
        mask = df_day["product_code"].isin(tickers)
        df_filtered = df_day[mask].copy()

        if df_filtered.empty:
            time.sleep(REQUEST_DELAY)
            continue

        df_filtered["date"] = pd.Timestamp(report_date)
        all_records.append(df_filtered)
        time.sleep(REQUEST_DELAY)

    log.info(f"Download complete: ok={n_ok} 404={n_404} err={n_err}")

    if not all_records:
        log.error("No data fetched — output not written")
        sys.exit(1)

    # Build new data
    df_new = pd.concat(all_records, ignore_index=True)
    df_new["date"] = pd.to_datetime(df_new["date"])

    # Merge with existing if incremental
    if args.incremental and out_path.exists():
        df_existing = pd.read_parquet(out_path)
        df_new = pd.concat([df_existing, df_new], ignore_index=True)

    # Deduplicate and sort
    df_new = (df_new
              .drop_duplicates(subset=["date", "product_code"])
              .sort_values(["product_code", "date"])
              .reset_index(drop=True))

    # Validate
    issues = []
    n_missing_pct = int(df_new["short_pct"].isna().sum())
    if n_missing_pct > 0:
        issues.append({"code": "MISSING_SHORT_PCT", "count": n_missing_pct})

    tickers_found   = set(df_new["product_code"].unique())
    tickers_missing = tickers - tickers_found
    if tickers_missing:
        issues.append({
            "code":    "TICKERS_NOT_IN_ASIC",
            "tickers": sorted(tickers_missing),
            "note":    "These tickers never appeared in any ASIC report — "
                       "likely not shortable or listed after fetch window"
        })

    # Save
    df_new.to_parquet(out_path, index=False)
    log.info(f"Saved {len(df_new)} rows → {out_path}")

    # Coverage stats per ticker
    ticker_stats = (df_new.groupby("product_code")
                    .agg(
                        rows=("date", "count"),
                        date_min=("date", "min"),
                        date_max=("date", "max"),
                        avg_short_pct=("short_pct", "mean"),
                    )
                    .reset_index()
                    .sort_values("product_code"))

    # Summary
    summary = {
        "generated_at":   datetime.utcnow().isoformat(),
        "fetch_start":    str(fetch_start),
        "fetch_end":      str(fetch_end),
        "days_attempted": len(days_to_fetch),
        "days_ok":        n_ok,
        "days_404":       n_404,
        "days_error":     n_err,
        "total_rows":     len(df_new),
        "tickers_found":  len(tickers_found),
        "tickers_missing": sorted(tickers_missing),
        "issues":         issues,
        "ticker_coverage": ticker_stats.to_dict(orient="records"),
    }

    summary_path = out_dir / "_fetch_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)

    log.info("=" * 60)
    log.info("FETCH COMPLETE")
    log.info(f"  Rows        : {len(df_new)}")
    log.info(f"  Tickers     : {len(tickers_found)} / {len(tickers)}")
    log.info(f"  Days OK     : {n_ok}")
    log.info(f"  Days 404    : {n_404}")
    log.info(f"  Issues      : {len(issues)}")
    log.info(f"  Summary     → {summary_path}")
    log.info("=" * 60)

    if tickers_missing:
        log.warning(f"Tickers not found in any ASIC report: {sorted(tickers_missing)}")

    log.info("Short interest fetch complete. Batch 5 done.")


if __name__ == "__main__":
    main()