#!/usr/bin/env python3
"""
fetch_companies.py
==================
Fetches ASX 50 OHLCV data from yfinance.

Commercial-grade features:
  - auto_adjust=False: raw close and adjusted close stored separately
  - actions=True: dividends and splits fetched for corporate action audit
  - Point-in-time fetch: each ticker fetched from its list_date
  - Return validation: impossible returns quarantined as NaN, never deleted
  - Per-ticker manifest: yfinance version, row count, date range, issues
  - Pipeline-level summary: exit code 1 on any hard failure

Usage:
    python -m scripts.fetch.equity.fetch_companies

Output:
    data/raw/companies/<TICKER>.parquet
    data/raw/companies/<TICKER>_manifest.json
    data/raw/companies/_fetch_summary.json
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

# ── Project root ──────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.utils.yf_utils import quarantine, validate, write_manifest
from scripts.utils.canonical_map import safe_name

load_dotenv(PROJECT_ROOT / ".env")

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(PROJECT_ROOT / "logs" / "fetch_companies.log"),
    ],
)
log = logging.getLogger(__name__)

CONFIG_PATH   = PROJECT_ROOT / "config" / "data.yaml"
MANIFEST_PATH = PROJECT_ROOT / "config" / "universe_manifest.json"


# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────

def load_config() -> dict:
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


def load_universe_manifest() -> dict:
    with open(MANIFEST_PATH) as f:
        return json.load(f)


# ─────────────────────────────────────────────────────────────────────────────
# Clean
# ─────────────────────────────────────────────────────────────────────────────

def clean_yf(df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()

    df = df.copy().reset_index()

    # Flatten MultiIndex columns from yf.download()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] if c[1] in ("", ticker) else f"{c[0]}_{c[1]}"
                      for c in df.columns]

    df.columns = [str(c).lower().strip().replace(" ", "_") for c in df.columns]

    # Rename adj_close → close_adj
    df = df.rename(columns={"adj_close": "close_adj"})

    date_col = next((c for c in df.columns if "date" in c), None)
    if date_col is None:
        return pd.DataFrame()

    df["date"] = pd.to_datetime(df[date_col], errors="coerce")
    df = (df.dropna(subset=["date"])
            .sort_values("date")
            .drop_duplicates("date")
            .reset_index(drop=True))

    keep = ["date", "open", "high", "low", "close", "close_adj",
            "volume", "dividends", "stock_splits"]
    return df[[c for c in keep if c in df.columns]]


# ─────────────────────────────────────────────────────────────────────────────
# Fetch one ticker
# ─────────────────────────────────────────────────────────────────────────────

def fetch_one(ticker: str, start: str, end: str, out_dir: Path) -> dict:

    log.info(f"Fetching {ticker} ...")

    try:
        raw = yf.download(
            ticker,
            start=start,
            end=end,
            auto_adjust=False,
            actions=True,
            progress=False,
        )
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
        log.warning(f"[{ticker}] Empty after cleaning")
        return {
            "ticker": ticker, "status": "EMPTY_AFTER_CLEAN",
            "issues": [{"code": "EMPTY_AFTER_CLEAN"}],
            "rows": 0, "date_min": None, "date_max": None,
        }

    # Validate and quarantine
    issues = validate(df, ticker, instrument_type="equity")
    df     = quarantine(df)

    hard_fails = [i for i in issues if i["code"] == "NON_POSITIVE_PRICE"]
    status = "FAIL" if hard_fails else "WARN" if issues else "OK"

    dest = out_dir / "quarantine" if status == "FAIL" else out_dir
    dest.mkdir(parents=True, exist_ok=True)

    out_path = dest / f"{safe_name(ticker)}.parquet"
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


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    cfg      = load_config()
    universe = load_universe_manifest()

    global_start: str = cfg["data"]["start_date"]
    global_end:   str = cfg["data"]["end_date"] or date.today().isoformat()
    out_dir = PROJECT_ROOT / cfg["raw"]["companies"]
    out_dir.mkdir(parents=True, exist_ok=True)

    tickers: dict = cfg["companies"]["tickers"]

    log.info("=" * 60)
    log.info("fetch_companies.py — ASX 50 OHLCV fetch")
    log.info(f"Universe   : {len(tickers)} tickers")
    log.info(f"Date range : {global_start} → {global_end}")
    log.info(f"Output     : {out_dir}")
    log.info(f"yfinance   : {yf.__version__}")
    log.info("=" * 60)

    results = []

    for ticker, sector in tickers.items():

        meta = universe.get("tickers", {}).get(ticker, {})
        list_date = meta.get("list_date", global_start)
        effective_start = max(
            pd.Timestamp(global_start),
            pd.Timestamp(list_date),
        ).strftime("%Y-%m-%d")

        if effective_start != global_start:
            log.info(f"[{ticker}] Partial-history — fetching from {effective_start}")

        result = fetch_one(ticker, effective_start, global_end, out_dir)
        result["sector"]  = sector
        result["history"] = meta.get("history", "unknown")
        results.append(result)

    # ── Summary ───────────────────────────────────────────────────────────────
    n_ok   = sum(1 for r in results if r["status"] == "OK")
    n_warn = sum(1 for r in results if r["status"] == "WARN")
    n_fail = sum(1 for r in results if r["status"] == "FAIL")
    n_err  = sum(1 for r in results if r["status"] in
                 {"NO_DATA", "DOWNLOAD_ERROR", "EMPTY_AFTER_CLEAN"})

    summary = {
        "generated_at":     datetime.utcnow().isoformat(),
        "yfinance_version": yf.__version__,
        "global_start":     global_start,
        "global_end":       global_end,
        "total":            len(results),
        "ok":               n_ok,
        "warn":             n_warn,
        "fail":             n_fail,
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
    log.info(f"  FAIL  : {n_fail}")
    log.info(f"  ERROR : {n_err}")
    log.info(f"  Summary → {summary_path}")
    log.info("=" * 60)

    failed  = [r["ticker"] for r in results if r["status"] == "FAIL"]
    errored = [r["ticker"] for r in results
               if r["status"] in {"NO_DATA", "DOWNLOAD_ERROR"}]

    if failed:
        log.error(f"PIPELINE FAIL — quarantined: {failed}")
        sys.exit(1)
    if errored:
        log.error(f"PIPELINE ERROR — no data: {errored}")
        sys.exit(1)

    log.info("All tickers fetched successfully. Ready for clean stage.")


if __name__ == "__main__":
    main()