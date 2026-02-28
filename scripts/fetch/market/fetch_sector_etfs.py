#!/usr/bin/env python3
"""
fetch_sector_etfs.py
====================
Fetches US sector ETFs from yfinance.

ETFs: XLF, XLK, XLI, XLE, XLV, XLU, XLB, XLP, XLY, XLRE, XLC, GDX

Note on partial history:
  XLRE listed 2015-10-08 — expected ~2700 rows
  XLC  listed 2018-06-19 — expected ~1950 rows
  GDX  listed 2006-05-22 — expected ~5000 rows
  All others full history from 2005.

Usage:
    python -m scripts.fetch.market.fetch_sector_etfs

Output:
    data/raw/sector/<TICKER>.parquet
    data/raw/sector/<TICKER>_manifest.json
    data/raw/sector/_fetch_summary.json
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

load_dotenv(PROJECT_ROOT / ".env")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(PROJECT_ROOT / "logs" / "fetch_sector_etfs.log"),
    ],
)
log = logging.getLogger(__name__)

CONFIG_PATH   = PROJECT_ROOT / "config" / "data.yaml"
MANIFEST_PATH = PROJECT_ROOT / "config" / "universe_manifest.json"

# Known partial-history ETFs with their list dates
ETF_LIST_DATES = {
    "XLRE": "2015-10-08",
    "XLC":  "2018-06-19",
    "GDX":  "2006-05-22",
}


def load_config() -> dict:
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


def load_universe_manifest() -> dict:
    with open(MANIFEST_PATH) as f:
        return json.load(f)


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


def fetch_one(
    ticker: str,
    start: str,
    end: str,
    out_dir: Path,
) -> dict:

    log.info(f"Fetching {ticker} ...")

    try:
        raw = yf.download(ticker, start=start, end=end,
                          auto_adjust=False, actions=False, progress=False)
    except Exception as exc:
        log.error(f"[{ticker}] Download error: {exc}")
        return {
            "ticker": ticker, "status": "DOWNLOAD_ERROR",
            "issues": [{"code": "DOWNLOAD_ERROR", "detail": str(exc)}],
            "rows": 0, "date_min": None, "date_max": None,
        }

    if raw is None or raw.empty:
        log.warning(f"[{ticker}] No data returned")
        return {
            "ticker": ticker, "status": "NO_DATA",
            "issues": [{"code": "NO_DATA"}],
            "rows": 0, "date_min": None, "date_max": None,
        }

    df = clean_yf(raw, ticker)

    if df.empty:
        return {
            "ticker": ticker, "status": "EMPTY_AFTER_CLEAN",
            "issues": [{"code": "EMPTY_AFTER_CLEAN"}],
            "rows": 0, "date_min": None, "date_max": None,
        }

    issues = validate(df, ticker, instrument_type="etf")
    df     = quarantine(df)

    status = "WARN" if issues else "OK"

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{ticker}.parquet"
    df.to_parquet(out_path, index=False)

    write_manifest(str(out_path), ticker, df, issues, status)

    log.info(
        f"[{ticker}] {status} | rows={len(df)} | "
        f"{df['date'].min().date()} → {df['date'].max().date()} | "
        f"issues={len(issues)}"
    )

    return {
        "ticker":   ticker,
        "status":   status,
        "issues":   issues,
        "rows":     len(df),
        "date_min": str(df["date"].min().date()),
        "date_max": str(df["date"].max().date()),
        "output":   str(out_path),
    }


def main() -> None:
    cfg = load_config()

    global_start: str = cfg["data"]["start_date"]
    end:          str = cfg["data"]["end_date"] or date.today().isoformat()
    out_dir = PROJECT_ROOT / cfg["raw"]["sector"]
    out_dir.mkdir(parents=True, exist_ok=True)

    tickers: list = cfg["sector"]["tickers"]

    log.info("=" * 60)
    log.info("fetch_sector_etfs.py — US sector ETFs")
    log.info(f"Tickers    : {tickers}")
    log.info(f"Date range : {global_start} → {end}")
    log.info(f"yfinance   : {yf.__version__}")
    log.info("=" * 60)

    results = []

    for ticker in tickers:
        # Use list date for partial-history ETFs
        list_date = ETF_LIST_DATES.get(ticker, global_start)
        effective_start = max(
            pd.Timestamp(global_start),
            pd.Timestamp(list_date),
        ).strftime("%Y-%m-%d")

        if effective_start != global_start:
            log.info(f"[{ticker}] Partial-history — fetching from {effective_start}")

        result = fetch_one(ticker, effective_start, end, out_dir)
        result["list_date"] = list_date
        results.append(result)

    # ── Summary ───────────────────────────────────────────────────────────────
    n_ok   = sum(1 for r in results if r["status"] == "OK")
    n_warn = sum(1 for r in results if r["status"] == "WARN")
    n_err  = sum(1 for r in results if r["status"] not in {"OK", "WARN"})

    summary = {
        "generated_at":     datetime.utcnow().isoformat(),
        "yfinance_version": yf.__version__,
        "global_start":     global_start,
        "end":              end,
        "total":            len(results),
        "ok":               n_ok,
        "warn":             n_warn,
        "error":            n_err,
        "tickers":          results,
    }

    summary_path = out_dir / "_fetch_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    log.info("=" * 60)
    log.info("FETCH COMPLETE")
    log.info(f"  OK    : {n_ok}")
    log.info(f"  WARN  : {n_warn}")
    log.info(f"  ERROR : {n_err}")
    log.info(f"  Summary → {summary_path}")
    log.info("=" * 60)

    errored = [r["ticker"] for r in results
               if r["status"] not in {"OK", "WARN"}]
    if errored:
        log.error(f"PIPELINE ERROR — no data: {errored}")
        sys.exit(1)

    log.info("All sector ETFs fetched. Ready for next script.")


if __name__ == "__main__":
    main()
