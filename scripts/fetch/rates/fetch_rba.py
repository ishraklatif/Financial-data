#!/usr/bin/env python3
"""
fetch_rba.py
============
Processes manually downloaded RBA CSV files into clean parquet outputs.

Place all files in data/raw/rba/downloads/ before running.

Files processed (all from www.rba.gov.au/statistics/tables/):
  f1-data.csv  → CASH_RATE (FIRMMCRTD), daily
  f2-data.csv  → YIELD_2Y (FCMYGBAG2D), YIELD_10Y (FCMYGBAG10D), daily
  g1-data.csv  → CPI_INDEX (GCPIAG), CPI_YOY (GCPIAGYP), quarterly
  h5-data.csv  → UNEMP (GLFSURSA), monthly
  f3-data.csv  → F3 corporate bond yields (A and BBB, 4 tenors), monthly

Note on F1 history:
  FIRMMCRTD starts 2011-01-04. Pre-2011 cash rate is not in this file.

Usage:
    python -m scripts.fetch.rates.fetch_rba

Output:
    data/raw/rba/rates/CASH_RATE.parquet
    data/raw/rba/rates/YIELD_2Y.parquet
    data/raw/rba/rates/YIELD_10Y.parquet
    data/raw/rba/macro/CPI_INDEX.parquet
    data/raw/rba/macro/CPI_YOY.parquet
    data/raw/rba/macro/UNEMP.parquet
    data/raw/rba/credit/F3_*.parquet  (8 files)
    data/raw/rba/_fetch_summary.json
"""

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
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
        logging.FileHandler(PROJECT_ROOT / "logs" / "fetch_rba.log"),
    ],
)
log = logging.getLogger(__name__)

CONFIG_PATH   = PROJECT_ROOT / "config" / "data.yaml"
DOWNLOADS_DIR = PROJECT_ROOT / "data" / "raw" / "rba" / "downloads"

DEFAULT_FILES = {
    "f1": "f1-data.csv",
    "f2": "f2-data.csv",
    "g1": "g1-data.csv",
    "h5": "h5-data.csv",
    "f3": "f3-data.csv",
}

F3_SERIES = [
    ("F3_A_YIELD_3Y",    "FNFYA3M",    "A-rated NFC bond yield — 3Y tenor"),
    ("F3_A_YIELD_5Y",    "FNFYA5M",    "A-rated NFC bond yield — 5Y tenor"),
    ("F3_A_YIELD_7Y",    "FNFYA7M",    "A-rated NFC bond yield — 7Y tenor"),
    ("F3_A_YIELD_10Y",   "FNFYA10M",   "A-rated NFC bond yield — 10Y tenor"),
    ("F3_BBB_YIELD_3Y",  "FNFYBBB3M",  "BBB-rated NFC bond yield — 3Y tenor"),
    ("F3_BBB_YIELD_5Y",  "FNFYBBB5M",  "BBB-rated NFC bond yield — 5Y tenor"),
    ("F3_BBB_YIELD_7Y",  "FNFYBBB7M",  "BBB-rated NFC bond yield — 7Y tenor"),
    ("F3_BBB_YIELD_10Y", "FNFYBBB10M", "BBB-rated NFC bond yield — 10Y tenor"),
]


def load_config() -> dict:
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


# ─────────────────────────────────────────────────────────────────────────────
# RBA CSV parser — windows-1252, wide format
# ─────────────────────────────────────────────────────────────────────────────

def parse_rba_csv(path: Path, series_id: str, canonical_name: str) -> pd.DataFrame:
    """
    Extract one series from a wide-format RBA CSV.
    Handles windows-1252 encoding and UTF-8 BOM.
    Date formats: DD-Mon-YYYY (F1/F2) or DD/MM/YYYY (G1/H5/F3).
    """
    with open(path, "rb") as f:
        raw = f.read()

    if raw[:3] == b"\xef\xbb\xbf":
        raw = raw[3:]
    text = raw.decode("windows-1252", errors="replace")
    lines = text.replace("\r\n", "\n").replace("\r", "\n").split("\n")

    series_row_idx = next(
        (i for i, l in enumerate(lines) if l.startswith("Series ID")), None
    )
    if series_row_idx is None:
        raise ValueError(f"No 'Series ID' row in {path.name}")

    series_ids = [s.strip() for s in lines[series_row_idx].split(",")]
    if series_id not in series_ids:
        available = [s for s in series_ids if s and s != "Series ID"]
        raise ValueError(
            f"'{series_id}' not found in {path.name}. Available: {available[:8]}"
        )
    col_idx = series_ids.index(series_id)

    records = []
    for line in lines[series_row_idx + 1:]:
        parts = line.split(",")
        if not parts or not parts[0].strip():
            continue
        date_str = parts[0].strip()

        dt = None
        for fmt in ("%d-%b-%Y", "%d/%m/%Y", "%Y-%m-%d"):
            try:
                dt = pd.to_datetime(date_str, format=fmt)
                break
            except Exception:
                continue
        if dt is None:
            continue

        val = float("nan")
        if col_idx < len(parts) and parts[col_idx].strip():
            try:
                val = float(parts[col_idx].strip())
            except (ValueError, TypeError):
                pass

        records.append({"date": dt, "value": val})

    if not records:
        raise ValueError(f"No data parsed for {series_id} in {path.name}")

    df = pd.DataFrame(records)
    df["series_id"]   = series_id
    df["series_name"] = canonical_name
    return (df.dropna(subset=["date"])
              .sort_values("date")
              .drop_duplicates("date")
              .reset_index(drop=True))


# ─────────────────────────────────────────────────────────────────────────────
# Save + manifest
# ─────────────────────────────────────────────────────────────────────────────

def save(canonical_name: str, description: str, df: pd.DataFrame,
         start: str, out_dir: Path) -> dict:

    df = df[df["date"] >= start].copy().reset_index(drop=True)

    issues = []
    n_missing = int(df["value"].isna().sum())
    if n_missing > 0:
        issues.append({"code": "MISSING_VALUES", "count": n_missing})
    if len(df) == 0:
        issues.append({"code": "EMPTY"})

    status = "WARN" if issues else "OK"

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{canonical_name}.parquet"
    df.to_parquet(out_path, index=False)

    manifest = {
        "canonical_name": canonical_name,
        "description":    description,
        "output":         str(out_path),
        "status":         status,
        "fetched_at":     datetime.utcnow().isoformat(),
        "rows":           len(df),
        "date_min":       str(df["date"].min().date()) if len(df) else None,
        "date_max":       str(df["date"].max().date()) if len(df) else None,
        "missing_values": n_missing,
        "issues":         issues,
    }
    with open(out_dir / f"{canonical_name}_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    log.info(
        f"[{canonical_name}] {status} | rows={len(df)} | missing={n_missing} | "
        f"{df['date'].min().date() if len(df) else 'N/A'} → "
        f"{df['date'].max().date() if len(df) else 'N/A'}"
    )

    return {
        "canonical_name": canonical_name,
        "description":    description,
        "status":         status,
        "issues":         issues,
        "rows":           len(df),
        "date_min":       str(df["date"].min().date()) if len(df) else None,
        "date_max":       str(df["date"].max().date()) if len(df) else None,
        "output":         str(out_path),
    }


def error_result(canonical_name: str, detail: str) -> dict:
    log.error(f"[{canonical_name}] ERROR: {detail}")
    return {
        "canonical_name": canonical_name,
        "status":         "ERROR",
        "issues":         [{"code": "ERROR", "detail": detail[:200]}],
        "rows":           0,
        "date_min":       None,
        "date_max":       None,
    }


def resolve(arg_val, key: str):
    if arg_val:
        p = Path(arg_val)
        return p if p.exists() else None
    default = DOWNLOADS_DIR / DEFAULT_FILES[key]
    return default if default.exists() else None


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--f1", default=None)
    parser.add_argument("--f2", default=None)
    parser.add_argument("--g1", default=None)
    parser.add_argument("--h5", default=None)
    parser.add_argument("--f3", default=None)
    args = parser.parse_args()

    cfg   = load_config()
    start = cfg["data"]["start_date"]

    rba_base   = PROJECT_ROOT / cfg["raw"]["rba_rates"].rsplit("/", 1)[0].rsplit("/", 1)[0]
    rates_dir  = PROJECT_ROOT / cfg["raw"]["rba_rates"]
    macro_dir  = PROJECT_ROOT / "data" / "raw" / "rba" / "macro"
    credit_dir = PROJECT_ROOT / cfg["raw"]["rba_credit"]

    log.info("=" * 60)
    log.info("fetch_rba.py — RBA manual file processor")
    log.info(f"Start date : {start}")
    log.info(f"Downloads  : {DOWNLOADS_DIR}")
    log.info("=" * 60)

    results = []

    # F1 — Cash rate
    f1 = resolve(args.f1, "f1")
    if f1:
        try:
            df = parse_rba_csv(f1, "FIRMMCRTD", "CASH_RATE")
            results.append(save("CASH_RATE", "RBA cash rate target (daily)", df, start, rates_dir))
        except Exception as e:
            results.append(error_result("CASH_RATE", str(e)))
    else:
        log.warning("F1 not found — skipping CASH_RATE")

    # F2 — Bond yields
    f2 = resolve(args.f2, "f2")
    if f2:
        for canonical, sid, desc in [
            ("YIELD_2Y",  "FCMYGBAG2D",  "AUS 2Y government bond yield (daily)"),
            ("YIELD_10Y", "FCMYGBAG10D", "AUS 10Y government bond yield (daily)"),
        ]:
            try:
                df = parse_rba_csv(f2, sid, canonical)
                results.append(save(canonical, desc, df, start, rates_dir))
            except Exception as e:
                results.append(error_result(canonical, str(e)))
    else:
        log.warning("F2 not found — skipping bond yields")

    # G1 — CPI
    g1 = resolve(args.g1, "g1")
    if g1:
        for canonical, sid, desc in [
            ("CPI_INDEX", "GCPIAG",   "AUS CPI all groups index (quarterly)"),
            ("CPI_YOY",   "GCPIAGYP", "AUS CPI year-on-year % (quarterly)"),
        ]:
            try:
                df = parse_rba_csv(g1, sid, canonical)
                results.append(save(canonical, desc, df, start, macro_dir))
            except Exception as e:
                results.append(error_result(canonical, str(e)))
    else:
        log.warning("G1 not found — skipping CPI")

    # H5 — Unemployment
    h5 = resolve(args.h5, "h5")
    if h5:
        try:
            df = parse_rba_csv(h5, "GLFSURSA", "UNEMP")
            results.append(save("UNEMP", "AUS unemployment rate SA (monthly)", df, start, macro_dir))
        except Exception as e:
            results.append(error_result("UNEMP", str(e)))
    else:
        log.warning("H5 not found — skipping unemployment")

    # F3 — Corporate bond yields
    f3 = resolve(args.f3, "f3")
    if f3:
        for canonical, sid, desc in F3_SERIES:
            try:
                df = parse_rba_csv(f3, sid, canonical)
                results.append(save(canonical, desc, df, start, credit_dir))
            except Exception as e:
                results.append(error_result(canonical, str(e)))
    else:
        log.warning("F3 not found — skipping corporate bond yields")

    # Summary
    n_ok   = sum(1 for r in results if r["status"] == "OK")
    n_warn = sum(1 for r in results if r["status"] == "WARN")
    n_err  = sum(1 for r in results if r["status"] == "ERROR")

    summary = {
        "generated_at": datetime.utcnow().isoformat(),
        "start":        start,
        "total":        len(results),
        "ok":           n_ok,
        "warn":         n_warn,
        "error":        n_err,
        "series":       results,
    }

    summary_dir = PROJECT_ROOT / "data" / "raw" / "rba"
    summary_dir.mkdir(parents=True, exist_ok=True)
    summary_path = summary_dir / "_fetch_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    log.info("=" * 60)
    log.info("FETCH COMPLETE")
    log.info(f"  OK    : {n_ok}")
    log.info(f"  WARN  : {n_warn}")
    log.info(f"  ERROR : {n_err}")
    log.info(f"  Summary → {summary_path}")
    log.info("=" * 60)

    if n_err > 0:
        errored = [r["canonical_name"] for r in results if r["status"] == "ERROR"]
        log.error(f"PIPELINE ERROR: {errored}")
        sys.exit(1)

    log.info("RBA fetch complete.")


if __name__ == "__main__":
    main()