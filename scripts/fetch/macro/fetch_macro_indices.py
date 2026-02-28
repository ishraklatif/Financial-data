#!/usr/bin/env python3
"""
fetch_macro_indices.py
======================
Fetches global macro indices and volatility indices from yfinance.

Indices:   AXJO, GSPC, FTSE, N225, HSI, SSE, CSI300
Volatility: VIX, VVIX, MOVE

Usage:
    python -m scripts.fetch.macro.fetch_macro_indices

Output:
    data/raw/macro/indices/<CANONICAL>.parquet
    data/raw/macro/indices/<CANONICAL>_manifest.json
    data/raw/macro/volatility/<CANONICAL>.parquet
    data/raw/macro/volatility/<CANONICAL>_manifest.json
    data/raw/macro/indices/_fetch_summary.json
"""

import json
import logging
import sys
from datetime import date, datetime
from pathlib import Path

import pandas as pd
import yaml
import yfinance as yf
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.utils.yf_utils import quarantine, validate, write_manifest
from scripts.utils.canonical_map import CANONICAL, canonical

load_dotenv(PROJECT_ROOT / ".env")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(PROJECT_ROOT / "logs" / "fetch_macro_indices.log"),
    ],
)
log = logging.getLogger(__name__)

CONFIG_PATH = PROJECT_ROOT / "config" / "data.yaml"


def load_config() -> dict:
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


# ─────────────────────────────────────────────────────────────────────────────
# Clean — same pattern as fetch_companies.py
# ─────────────────────────────────────────────────────────────────────────────

def clean_yf(df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()

    df = df.copy().reset_index()

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] if c[1] in ("", ticker) else f"{c[0]}_{c[1]}"
                      for c in df.columns]

    df.columns = [str(c).lower().strip().replace(" ", "_") for c in df.columns]
    df = df.rename(columns={"adj_close": "close_adj"})

    date_col = next((c for c in df.columns if "date" in c), None)
    if date_col is None:
        return pd.DataFrame()

    df["date"] = pd.to_datetime(df[date_col], errors="coerce")
    df = (df.dropna(subset=["date"])
            .sort_values("date")
            .drop_duplicates("date")
            .reset_index(drop=True))

    keep = ["date", "open", "high", "low", "close", "close_adj", "volume"]
    return df[[c for c in keep if c in df.columns]]


# ─────────────────────────────────────────────────────────────────────────────
# Fetch one ticker
# ─────────────────────────────────────────────────────────────────────────────

def fetch_one(
    ticker: str,
    canonical_name: str,
    instrument_type: str,
    start: str,
    end: str,
    out_dir: Path,
) -> dict:

    log.info(f"Fetching {ticker} → {canonical_name} ...")

    try:
        raw = yf.download(ticker, start=start, end=end,
                          auto_adjust=False, actions=False, progress=False)
    except Exception as exc:
        log.error(f"[{canonical_name}] Download error: {exc}")
        return {
            "ticker": ticker, "canonical": canonical_name,
            "status": "DOWNLOAD_ERROR",
            "issues": [{"code": "DOWNLOAD_ERROR", "detail": str(exc)}],
            "rows": 0, "date_min": None, "date_max": None,
        }

    if raw is None or raw.empty:
        log.warning(f"[{canonical_name}] No data returned")
        return {
            "ticker": ticker, "canonical": canonical_name,
            "status": "NO_DATA",
            "issues": [{"code": "NO_DATA"}],
            "rows": 0, "date_min": None, "date_max": None,
        }

    df = clean_yf(raw, ticker)

    if df.empty:
        log.warning(f"[{canonical_name}] Empty after cleaning")
        return {
            "ticker": ticker, "canonical": canonical_name,
            "status": "EMPTY_AFTER_CLEAN",
            "issues": [{"code": "EMPTY_AFTER_CLEAN"}],
            "rows": 0, "date_min": None, "date_max": None,
        }

    issues = validate(df, canonical_name, instrument_type=instrument_type)
    df     = quarantine(df)

    # Indices never have hard fails on price — no equity-style floor
    status = "WARN" if issues else "OK"

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{canonical_name}.parquet"
    df.to_parquet(out_path, index=False)

    write_manifest(str(out_path), canonical_name, df, issues, status)

    log.info(
        f"[{canonical_name}] {status} | rows={len(df)} | "
        f"{df['date'].min().date()} → {df['date'].max().date()} | "
        f"issues={len(issues)}"
    )

    return {
        "ticker":    ticker,
        "canonical": canonical_name,
        "status":    status,
        "issues":    issues,
        "rows":      len(df),
        "date_min":  str(df["date"].min().date()),
        "date_max":  str(df["date"].max().date()),
        "output":    str(out_path),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    cfg = load_config()

    start: str = cfg["data"]["start_date"]
    end:   str = cfg["data"]["end_date"] or date.today().isoformat()

    indices_dir = PROJECT_ROOT / cfg["raw"]["macro_indices"]
    vol_dir     = PROJECT_ROOT / cfg["raw"]["macro_vol"]

    # Build fetch list from config
    # Each entry: (raw_ticker, canonical_name, instrument_type, out_dir)
    fetch_list = []

    for canonical_name, ticker in cfg["macro"]["indices"].items():
        fetch_list.append((ticker, canonical_name, "index", indices_dir))

    for canonical_name, ticker in cfg["macro"]["volatility"].items():
        fetch_list.append((ticker, canonical_name, "volatility_index", vol_dir))

    log.info("=" * 60)
    log.info("fetch_macro_indices.py — global indices + volatility")
    log.info(f"Instruments : {len(fetch_list)}")
    log.info(f"Date range  : {start} → {end}")
    log.info(f"yfinance    : {yf.__version__}")
    log.info("=" * 60)

    results = []

    for ticker, canonical_name, instrument_type, out_dir in fetch_list:
        result = fetch_one(
            ticker, canonical_name, instrument_type, start, end, out_dir
        )
        result["instrument_type"] = instrument_type
        results.append(result)

    # ── Summary ───────────────────────────────────────────────────────────────
    n_ok   = sum(1 for r in results if r["status"] == "OK")
    n_warn = sum(1 for r in results if r["status"] == "WARN")
    n_err  = sum(1 for r in results if r["status"] not in {"OK", "WARN"})

    summary = {
        "generated_at":     datetime.utcnow().isoformat(),
        "yfinance_version": yf.__version__,
        "start":            start,
        "end":              end,
        "total":            len(results),
        "ok":               n_ok,
        "warn":             n_warn,
        "error":            n_err,
        "instruments":      results,
    }

    summary_path = indices_dir / "_fetch_summary.json"
    indices_dir.mkdir(parents=True, exist_ok=True)
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    log.info("=" * 60)
    log.info("FETCH COMPLETE")
    log.info(f"  OK    : {n_ok}")
    log.info(f"  WARN  : {n_warn}")
    log.info(f"  ERROR : {n_err}")
    log.info(f"  Summary → {summary_path}")
    log.info("=" * 60)

    errored = [r["canonical"] for r in results
               if r["status"] not in {"OK", "WARN"}]
    if errored:
        log.error(f"PIPELINE ERROR — no data: {errored}")
        sys.exit(1)

    log.info("All macro indices fetched. Ready for next script.")


if __name__ == "__main__":
    main()
