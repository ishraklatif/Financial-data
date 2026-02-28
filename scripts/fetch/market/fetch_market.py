#!/usr/bin/env python3
"""
fetch_market.py
===============
Fetches FX pairs, commodities, and DXY from yfinance.

FX:         AUDUSD, AUDJPY, AUDCNY, DXY
Commodities: GOLD, OIL (Brent), COPPER, SILVER, IRON, DBC
Additional:  BDI (Baltic Dry Index)

Key difference from equity fetch:
  - Commodity futures can have negative prices (WTI Apr 2020)
    → NON_POSITIVE_PRICE suppressed for commodity_future instrument type
  - Returns outside (-0.99, 5.0) are quarantined as NaN, price kept intact

Usage:
    python -m scripts.fetch.market.fetch_market

Output:
    data/raw/market/fx/<CANONICAL>.parquet
    data/raw/market/commodities/<CANONICAL>.parquet
    data/raw/market/commodities/<CANONICAL>_manifest.json
    data/raw/market/fx/_fetch_summary.json
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
from scripts.utils.canonical_map import canonical

load_dotenv(PROJECT_ROOT / ".env")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(PROJECT_ROOT / "logs" / "fetch_market.log"),
    ],
)
log = logging.getLogger(__name__)

CONFIG_PATH = PROJECT_ROOT / "config" / "data.yaml"


def load_config() -> dict:
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


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
    canonical_name: str,
    instrument_type: str,
    start: str,
    end: str,
    out_dir: Path,
) -> dict:

    log.info(f"Fetching {ticker} → {canonical_name} ({instrument_type}) ...")

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
        return {
            "ticker": ticker, "canonical": canonical_name,
            "status": "EMPTY_AFTER_CLEAN",
            "issues": [{"code": "EMPTY_AFTER_CLEAN"}],
            "rows": 0, "date_min": None, "date_max": None,
        }

    # Validate — NON_POSITIVE_PRICE suppressed for commodity futures and FX
    # (negative oil prices Apr 2020 are real, FX rates are always positive
    #  but we don't want the equity price floor logic applied)
    issues = validate(df, canonical_name, instrument_type=instrument_type)

    # For commodity_future: keep price intact, only NaN the return
    df = quarantine(df)

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
        "ticker":          ticker,
        "canonical":       canonical_name,
        "instrument_type": instrument_type,
        "status":          status,
        "issues":          issues,
        "rows":            len(df),
        "date_min":        str(df["date"].min().date()),
        "date_max":        str(df["date"].max().date()),
        "output":          str(out_path),
    }


def main() -> None:
    cfg = load_config()

    start: str = cfg["data"]["start_date"]
    end:   str = cfg["data"]["end_date"] or date.today().isoformat()

    fx_dir       = PROJECT_ROOT / cfg["raw"]["fx"]
    comm_dir     = PROJECT_ROOT / cfg["raw"]["commodities"]

    # Build fetch list from config
    # (raw_ticker, canonical_name, instrument_type, out_dir)
    fetch_list = []

    for canonical_name, ticker in cfg["market"]["fx"].items():
        itype = "fx_index" if canonical_name == "DXY" else "fx"
        fetch_list.append((ticker, canonical_name, itype, fx_dir))

    for canonical_name, ticker in cfg["market"]["commodities"].items():
        itype = "etf" if canonical_name == "DBC" else "commodity_future"
        fetch_list.append((ticker, canonical_name, itype, comm_dir))

    # Baltic Dry Index
    # bdi_ticker = cfg["additional"]["baltic_dry"]["ticker"]
    # fetch_list.append((bdi_ticker, "BDI", "index", comm_dir))

    log.info("=" * 60)
    log.info("fetch_market.py — FX, commodities, DXY, BDI")
    log.info(f"Instruments : {len(fetch_list)}")
    log.info(f"Date range  : {start} → {end}")
    log.info(f"yfinance    : {yf.__version__}")
    log.info("=" * 60)

    results = []

    for ticker, canonical_name, instrument_type, out_dir in fetch_list:
        result = fetch_one(ticker, canonical_name, instrument_type,
                           start, end, out_dir)
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

    summary_path = fx_dir / "_fetch_summary.json"
    fx_dir.mkdir(parents=True, exist_ok=True)
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

    log.info("All market data fetched. Ready for next script.")


if __name__ == "__main__":
    main()
