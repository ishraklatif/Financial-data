#!/usr/bin/env python3
"""
fetch_fred.py
=============
Fetches US macro data from the FRED API.

Series fetched:
  Inflation:  CPI (CPIAUCSL), PCE (PCEPI)
  Labour:     Unemployment (UNRATE), Job Openings (JTSJOL)
  Rates:      Fed Funds (FEDFUNDS), 2Y (DGS2), 10Y (DGS10),
              30Y (DGS30), 3M (DGS3MO)
  Growth:     Real GDP (GDPC1), Industrial Production (INDPRO),
              Mfg PMI proxy (IPMAN), Services PMI proxy (SRVPRD)
  Credit:     HY Spread (BAMLH0A0HYM2), IG Spread (BAMLCC0A1AAATRIV),
              TED Spread (TEDRATE) — ends 2022-01-21, documented
  Sentiment:  U Michigan Consumer Sentiment (UMCSENT)

Output schema per file:
  date, value, series_id, series_name, frequency

Usage:
    python -m scripts.fetch.rates.fetch_fred

Output:
    data/raw/fred/<SERIES_ID>.parquet
    data/raw/fred/<SERIES_ID>_manifest.json
    data/raw/fred/_fetch_summary.json
"""

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

# ── SSL fix for Mac — must be set before any HTTPS calls ─────────────────────
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
        logging.FileHandler(PROJECT_ROOT / "logs" / "fetch_fred.log"),
    ],
)
log = logging.getLogger(__name__)

CONFIG_PATH = PROJECT_ROOT / "config" / "data.yaml"

# Series discontinued before today — document, don't error
KNOWN_END_DATES = {
    "TEDRATE": "2022-01-21",
}

# Expected minimum rows per frequency
MIN_ROWS = {
    "daily":     4000,
    "weekly":    800,
    "monthly":   200,
    "quarterly": 60,
}


def load_config() -> dict:
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


# ─────────────────────────────────────────────────────────────────────────────
# Fetch one FRED series
# ─────────────────────────────────────────────────────────────────────────────

def fetch_one(
    series_id: str,
    series_name: str,
    start: str,
    end: str,
    out_dir: Path,
    fred,
) -> dict:

    log.info(f"Fetching FRED {series_id} ({series_name}) ...")

    # Adjust end date for discontinued series
    effective_end = KNOWN_END_DATES.get(series_id, end)
    if effective_end != end:
        log.info(f"[{series_id}] Discontinued — fetching up to {effective_end}")

    try:
        raw = fred.get_series(
            series_id,
            observation_start=start,
            observation_end=effective_end,
        )
    except Exception as exc:
        log.error(f"[{series_id}] FRED fetch error: {exc}")
        return {
            "series_id":   series_id,
            "series_name": series_name,
            "status":      "DOWNLOAD_ERROR",
            "issues":      [{"code": "DOWNLOAD_ERROR", "detail": str(exc)}],
            "rows":        0,
            "date_min":    None,
            "date_max":    None,
        }

    if raw is None or raw.empty:
        log.warning(f"[{series_id}] No data returned")
        return {
            "series_id":   series_id,
            "series_name": series_name,
            "status":      "NO_DATA",
            "issues":      [{"code": "NO_DATA"}],
            "rows":        0,
            "date_min":    None,
            "date_max":    None,
        }

    # ── Build clean DataFrame ─────────────────────────────────────────────────
    df = raw.reset_index()
    df.columns = ["date", "value"]
    df["date"]  = pd.to_datetime(df["date"], errors="coerce")
    df["value"] = pd.to_numeric(df["value"], errors="coerce")

    df = (df.dropna(subset=["date"])
            .sort_values("date")
            .drop_duplicates("date")
            .reset_index(drop=True))

    df["series_id"]   = series_id
    df["series_name"] = series_name

    # ── Detect frequency ──────────────────────────────────────────────────────
    if len(df) > 1:
        median_gap = df["date"].diff().median().days
        if median_gap <= 1:
            freq = "daily"
        elif median_gap <= 8:
            freq = "weekly"
        elif median_gap <= 35:
            freq = "monthly"
        else:
            freq = "quarterly"
    else:
        freq = "unknown"

    df["frequency"] = freq

    # ── Validate ──────────────────────────────────────────────────────────────
    issues = []
    n_missing = int(df["value"].isna().sum())

    if n_missing > 0:
        issues.append({"code": "MISSING_VALUES", "count": n_missing})

    min_rows = MIN_ROWS.get(freq, 50)
    if len(df) < min_rows:
        issues.append({
            "code":         "LOW_ROW_COUNT",
            "count":        len(df),
            "expected_min": min_rows,
            "frequency":    freq,
        })

    if series_id in KNOWN_END_DATES:
        issues.append({
            "code":     "SERIES_DISCONTINUED",
            "end_date": KNOWN_END_DATES[series_id],
            "note":     "Discontinued by FRED. Use replacement series post end_date.",
        })

    status = "WARN" if issues else "OK"

    # ── Save ──────────────────────────────────────────────────────────────────
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{series_id}.parquet"
    df.to_parquet(out_path, index=False)

    manifest = {
        "series_id":      series_id,
        "series_name":    series_name,
        "output":         str(out_path),
        "status":         status,
        "fetched_at":     datetime.utcnow().isoformat(),
        "frequency":      freq,
        "rows":           len(df),
        "date_min":       str(df["date"].min().date()),
        "date_max":       str(df["date"].max().date()),
        "missing_values": n_missing,
        "issues":         issues,
    }
    with open(out_dir / f"{series_id}_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    log.info(
        f"[{series_id}] {status} | rows={len(df)} | freq={freq} | "
        f"{df['date'].min().date()} → {df['date'].max().date()} | "
        f"missing={n_missing}"
    )

    return {
        "series_id":   series_id,
        "series_name": series_name,
        "status":      status,
        "issues":      issues,
        "frequency":   freq,
        "rows":        len(df),
        "date_min":    str(df["date"].min().date()),
        "date_max":    str(df["date"].max().date()),
        "output":      str(out_path),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    api_key = os.getenv("FRED_API_KEY")
    if not api_key or api_key == "your_fred_api_key_here":
        log.error(
            "FRED_API_KEY not set. "
            "Get a free key at: https://fred.stlouisfed.org/docs/api/api_key.html"
        )
        sys.exit(1)

    try:
        from fredapi import Fred
        fred = Fred(api_key=api_key)
    except ImportError:
        log.error("fredapi not installed. Run: pip install fredapi")
        sys.exit(1)

    cfg = load_config()

    start: str = cfg["data"]["start_date"]
    end:   str = cfg["data"]["end_date"] or date.today().isoformat()
    out_dir = PROJECT_ROOT / cfg["raw"]["fred"]

    fred_series: dict = cfg["fred"]["series"]

    log.info("=" * 60)
    log.info("fetch_fred.py — US macro via FRED API")
    log.info(f"Series     : {len(fred_series)}")
    log.info(f"Date range : {start} → {end}")
    log.info("=" * 60)

    results = []

    for series_name, series_id in fred_series.items():
        result = fetch_one(series_id, series_name, start, end, out_dir, fred)
        results.append(result)

    # ── Summary ───────────────────────────────────────────────────────────────
    n_ok   = sum(1 for r in results if r["status"] == "OK")
    n_warn = sum(1 for r in results if r["status"] == "WARN")
    n_err  = sum(1 for r in results if r["status"] not in {"OK", "WARN"})

    summary = {
        "generated_at": datetime.utcnow().isoformat(),
        "start":        start,
        "end":          end,
        "total":        len(results),
        "ok":           n_ok,
        "warn":         n_warn,
        "error":        n_err,
        "series":       results,
    }

    summary_path = out_dir / "_fetch_summary.json"
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    log.info("=" * 60)
    log.info("FETCH COMPLETE")
    log.info(f"  OK    : {n_ok}")
    log.info(f"  WARN  : {n_warn}")
    log.info(f"  ERROR : {n_err}")
    log.info(f"  Summary → {summary_path}")
    log.info("=" * 60)

    errored = [r["series_id"] for r in results
               if r["status"] not in {"OK", "WARN"}]
    if errored:
        log.error(f"PIPELINE ERROR — failed series: {errored}")
        sys.exit(1)

    log.info("All FRED series fetched. Ready for RBA fetch.")


if __name__ == "__main__":
    main()