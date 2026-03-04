#!/usr/bin/env python3
"""
fetch_rba.py
============
Processes RBA statistical tables into clean parquet outputs.

Supports two modes:
  1. AUTO: Downloads CSVs directly from the RBA website (preferred)
  2. MANUAL: Reads from data/raw/rba/downloads/ (fallback if RBA URL changes)

The script tries AUTO first and falls back to MANUAL automatically.

RBA URL structure (as of Feb 2026):
  Base: https://www.rba.gov.au/statistics/tables/csv/
  Files: f1-data.csv, f2-data.csv, f3-data.csv, g1-data.csv, h5-data.csv

  NOTE: RBA has previously served these at /statistics/tables/<name>.csv
  (without the csv/ subdirectory). If downloads fail, the script will try
  both URL patterns before falling back to manual files.

Files processed:
  f1-data.csv  → CASH_RATE      (FIRMMCRTD, daily, starts 2011-01-04)
  f2-data.csv  → YIELD_2Y       (FCMYGBAG2D, daily, starts 2013-05-20)
                 YIELD_10Y      (FCMYGBAG10D, daily, starts 2013-05-20)
  g1-data.csv  → CPI_INDEX      (GCPIAG, quarterly)
                 CPI_YOY        (GCPIAGYP, quarterly)
  h5-data.csv  → UNEMP          (GLFSURSA, monthly)
  f3-data.csv  → F3_A/BBB corporate bond yields (8 series, monthly)

NOTE on date coverage:
  CASH_RATE starts 2011-01-04 — this is a hard RBA data availability
  constraint. FIRMMCRTD is not available in F1 before 2011.
  Use patch_rba_cash_rate.py to extend back to 2005 from the RBA
  historical cash rate archive.

  YIELD_2Y / YIELD_10Y start 2013-05-20 — hard RBA constraint.
  These daily bond yield series were not published before May 2013.

Usage:
    python -m scripts.fetch.rates.fetch_rba
    python -m scripts.fetch.rates.fetch_rba --manual   # force manual mode
    python -m scripts.fetch.rates.fetch_rba --f1 /path/to/f1-data.csv

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
import time
from datetime import datetime
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
        logging.FileHandler(PROJECT_ROOT / "logs" / "fetch_rba.log"),
    ],
)
log = logging.getLogger(__name__)

CONFIG_PATH   = PROJECT_ROOT / "config" / "data.yaml"
DOWNLOADS_DIR = PROJECT_ROOT / "data" / "raw" / "rba" / "downloads"

# RBA URL patterns — tried in order until one succeeds
RBA_URL_PATTERNS = [
    "https://www.rba.gov.au/statistics/tables/csv/{filename}",
    "https://www.rba.gov.au/statistics/tables/{filename}",
]

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

REQUEST_TIMEOUT = 30
REQUEST_DELAY   = 1.0  # seconds between RBA downloads — be polite


def load_config() -> dict:
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


# ─────────────────────────────────────────────────────────────────────────────
# Download helpers
# ─────────────────────────────────────────────────────────────────────────────

def download_rba_csv(filename: str, session: requests.Session) -> bytes | None:
    """
    Try each URL pattern in order. Return raw bytes on success, None on failure.
    """
    for pattern in RBA_URL_PATTERNS:
        url = pattern.format(filename=filename)
        try:
            log.info(f"  Trying: {url}")
            resp = session.get(url, timeout=REQUEST_TIMEOUT)
            if resp.status_code == 200 and len(resp.content) > 1000:
                log.info(f"  ✓ Downloaded {filename} ({len(resp.content):,} bytes) from {url}")
                return resp.content
            else:
                log.warning(f"  ✗ {url} → HTTP {resp.status_code}")
        except requests.RequestException as exc:
            log.warning(f"  ✗ {url} → {exc}")
        time.sleep(0.5)
    return None


def resolve_file(arg_val: str | None, key: str,
                 auto_content: bytes | None) -> bytes | None:
    """
    Return file content in priority order:
      1. Explicit CLI argument path
      2. Auto-downloaded bytes
      3. Manual download directory
    """
    if arg_val:
        p = Path(arg_val)
        if p.exists():
            log.info(f"Using CLI-supplied file: {p}")
            return p.read_bytes()
        else:
            log.warning(f"CLI path not found: {p}")

    if auto_content is not None:
        return auto_content

    manual = DOWNLOADS_DIR / DEFAULT_FILES[key]
    if manual.exists():
        log.info(f"Using manual download: {manual}")
        return manual.read_bytes()

    return None


# ─────────────────────────────────────────────────────────────────────────────
# RBA CSV parser
# ─────────────────────────────────────────────────────────────────────────────

def parse_rba_csv(content: bytes, series_id: str,
                  canonical_name: str) -> pd.DataFrame:
    """
    Parse a wide-format RBA CSV (windows-1252 encoded).
    Handles both DD-Mon-YYYY (F1/F2) and DD/MM/YYYY (G1/H5/F3) date formats.
    Raises ValueError if series_id not found or no data parsed.
    """
    if content[:3] == b"\xef\xbb\xbf":
        content = content[3:]
    text = content.decode("windows-1252", errors="replace")
    lines = text.replace("\r\n", "\n").replace("\r", "\n").split("\n")

    sid_row = next(
        (i for i, l in enumerate(lines) if l.startswith("Series ID")), None
    )
    if sid_row is None:
        raise ValueError("No 'Series ID' row found in CSV")

    series_ids = [s.strip() for s in lines[sid_row].split(",")]
    if series_id not in series_ids:
        available = [s for s in series_ids if s and s != "Series ID"]
        raise ValueError(
            f"'{series_id}' not found. Available: {available[:10]}"
        )
    col_idx = series_ids.index(series_id)

    records = []
    for line in lines[sid_row + 1:]:
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
        raise ValueError(f"No data rows parsed for series_id='{series_id}'")

    return (
        pd.DataFrame(records)
        .assign(series_id=series_id, series_name=canonical_name)
        .dropna(subset=["date"])
        .sort_values("date")
        .drop_duplicates("date")
        .reset_index(drop=True)
    )


# ─────────────────────────────────────────────────────────────────────────────
# Save + manifest
# ─────────────────────────────────────────────────────────────────────────────

def save(canonical_name: str, description: str, df: pd.DataFrame,
         start: str, out_dir: Path) -> dict:
    df = df[df["date"] >= start].copy().reset_index(drop=True)

    issues    = []
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
        "issues":         [{"code": "ERROR", "detail": detail[:300]}],
        "rows":           0,
        "date_min":       None,
        "date_max":       None,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manual", action="store_true",
                        help="Skip auto-download, use files in rba/downloads/ only")
    parser.add_argument("--f1", default=None, help="Path to f1-data.csv")
    parser.add_argument("--f2", default=None, help="Path to f2-data.csv")
    parser.add_argument("--g1", default=None, help="Path to g1-data.csv")
    parser.add_argument("--h5", default=None, help="Path to h5-data.csv")
    parser.add_argument("--f3", default=None, help="Path to f3-data.csv")
    args = parser.parse_args()

    cfg   = load_config()
    start = cfg["data"]["start_date"]

    rates_dir  = PROJECT_ROOT / cfg["raw"]["rba_rates"]
    macro_dir  = PROJECT_ROOT / "data" / "raw" / "rba" / "macro"
    credit_dir = PROJECT_ROOT / cfg["raw"]["rba_credit"]

    log.info("=" * 60)
    log.info("fetch_rba.py — RBA statistical tables")
    log.info(f"Start date : {start}")
    log.info(f"Mode       : {'manual' if args.manual else 'auto (with manual fallback)'}")
    log.info("=" * 60)

    # ── Step 1: Auto-download if not in manual mode ───────────────────────────
    auto = {k: None for k in DEFAULT_FILES}

    if not args.manual:
        session = requests.Session()
        session.headers["User-Agent"] = (
            "Mozilla/5.0 (research data pipeline; contact: research@example.com)"
        )
        for key, filename in DEFAULT_FILES.items():
            log.info(f"Auto-downloading {filename} ...")
            auto[key] = download_rba_csv(filename, session)
            if auto[key]:
                time.sleep(REQUEST_DELAY)
            else:
                log.warning(
                    f"Auto-download failed for {filename}. "
                    f"Will try manual file at {DOWNLOADS_DIR / filename}"
                )

    # ── Step 2: Resolve final content per file ────────────────────────────────
    content = {
        "f1": resolve_file(args.f1, "f1", auto["f1"]),
        "f2": resolve_file(args.f2, "f2", auto["f2"]),
        "g1": resolve_file(args.g1, "g1", auto["g1"]),
        "h5": resolve_file(args.h5, "h5", auto["h5"]),
        "f3": resolve_file(args.f3, "f3", auto["f3"]),
    }

    for key, val in content.items():
        status = "OK" if val else "MISSING"
        log.info(f"  {DEFAULT_FILES[key]}: {status}")

    # ── Step 3: Parse and save ────────────────────────────────────────────────
    results = []

    # F1 — Cash rate (starts 2011 — hard RBA constraint)
    if content["f1"]:
        try:
            df = parse_rba_csv(content["f1"], "FIRMMCRTD", "CASH_RATE")
            result = save("CASH_RATE",
                          "RBA cash rate target (daily, starts 2011-01-04). "
                          "Run patch_rba_cash_rate.py to extend to 2005.",
                          df, start, rates_dir)
            results.append(result)
        except Exception as e:
            results.append(error_result("CASH_RATE", str(e)))
    else:
        log.warning("F1 unavailable — skipping CASH_RATE")

    # F2 — Bond yields (starts 2013 — hard RBA constraint)
    if content["f2"]:
        for canonical, sid, desc in [
            ("YIELD_2Y",  "FCMYGBAG2D",  "AUS 2Y government bond yield (daily, starts 2013-05-20)"),
            ("YIELD_10Y", "FCMYGBAG10D", "AUS 10Y government bond yield (daily, starts 2013-05-20)"),
        ]:
            try:
                df = parse_rba_csv(content["f2"], sid, canonical)
                results.append(save(canonical, desc, df, start, rates_dir))
            except Exception as e:
                results.append(error_result(canonical, str(e)))
    else:
        log.warning("F2 unavailable — skipping bond yields")

    # G1 — CPI (quarterly, full history from 2005)
    if content["g1"]:
        for canonical, sid, desc in [
            ("CPI_INDEX", "GCPIAG",   "AUS CPI all groups index (quarterly)"),
            ("CPI_YOY",   "GCPIAGYP", "AUS CPI year-on-year % (quarterly)"),
        ]:
            try:
                df = parse_rba_csv(content["g1"], sid, canonical)
                results.append(save(canonical, desc, df, start, macro_dir))
            except Exception as e:
                results.append(error_result(canonical, str(e)))
    else:
        log.warning("G1 unavailable — skipping CPI")

    # H5 — Unemployment (monthly, full history)
    if content["h5"]:
        try:
            df = parse_rba_csv(content["h5"], "GLFSURSA", "UNEMP")
            results.append(save("UNEMP",
                                "AUS unemployment rate SA (monthly)", df, start, macro_dir))
        except Exception as e:
            results.append(error_result("UNEMP", str(e)))
    else:
        log.warning("H5 unavailable — skipping unemployment")

    # F3 — Corporate bond yields (monthly, full history from 2005)
    if content["f3"]:
        for canonical, sid, desc in F3_SERIES:
            try:
                df = parse_rba_csv(content["f3"], sid, canonical)
                results.append(save(canonical, desc, df, start, credit_dir))
            except Exception as e:
                results.append(error_result(canonical, str(e)))
    else:
        log.warning("F3 unavailable — skipping corporate bond yields")

    # ── Summary ───────────────────────────────────────────────────────────────
    n_ok   = sum(1 for r in results if r["status"] == "OK")
    n_warn = sum(1 for r in results if r["status"] == "WARN")
    n_err  = sum(1 for r in results if r["status"] == "ERROR")

    summary = {
        "generated_at": datetime.utcnow().isoformat(),
        "start":        start,
        "mode":         "manual" if args.manual else "auto_with_fallback",
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
    if any(r["canonical_name"] == "CASH_RATE" and r["status"] in ("OK", "WARN")
           for r in results):
        log.info(
            "NOTE: CASH_RATE starts 2011-01-04 (hard RBA F1 constraint). "
            "Run patch_rba_cash_rate.py to extend to 2005."
        )


if __name__ == "__main__":
    main()
