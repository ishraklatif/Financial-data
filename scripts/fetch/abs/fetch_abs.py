#!/usr/bin/env python3
"""
fetch_abs.py
============
Processes manually downloaded ABS files into clean parquet outputs.

Place all files in data/raw/abs/downloads/ before running.

Files processed:
  all.xml       → GDP, GDP_PCA, HSR, TOT  (from ANA_AGG dataflow, quarterly)
  634501.xlsx   → WPI                     (ABS cat 6345.0, quarterly)

The ABS API (data.api.abs.gov.au) blocks automated requests with 403.
These files must be downloaded manually:

  all.xml:
    https://data.api.abs.gov.au/rest/data/ANA_AGG/ABS/
    (or via ABS Data Explorer — National Accounts)

  634501.xlsx:
    https://www.abs.gov.au/statistics/economy/price-indexes-and-inflation/
    wage-price-index-australia/latest-release
    → Table 1: Total hourly rates of pay excluding bonuses, sector, SA

Usage:
    python -m scripts.fetch.abs.fetch_abs

Output:
    data/raw/abs/gdp/GDP.parquet
    data/raw/abs/gdp/GDP_PCA.parquet
    data/raw/abs/gdp/HSR.parquet
    data/raw/abs/gdp/TOT.parquet
    data/raw/abs/wpi/WPI.parquet
    data/raw/abs/_fetch_summary.json
"""

import argparse
import json
import logging
import sys
import xml.etree.ElementTree as ET
from datetime import datetime
from pathlib import Path

import openpyxl
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
        logging.FileHandler(PROJECT_ROOT / "logs" / "fetch_abs.log"),
    ],
)
log = logging.getLogger(__name__)

CONFIG_PATH   = PROJECT_ROOT / "config" / "data.yaml"
DOWNLOADS_DIR = PROJECT_ROOT / "data" / "raw" / "abs" / "downloads"

DEFAULT_FILES = {
    "xml": "all.xml",
    "wpi": "634501.xlsx",
}

# SDMX namespaces for all.xml
NS = {
    "generic": "http://www.sdmx.org/resources/sdmxml/schemas/v2_1/data/generic",
    "message": "http://www.sdmx.org/resources/sdmxml/schemas/v2_1/message",
}

# Series to extract from all.xml (ANA_AGG)
# (canonical_name, MEASURE, DATA_ITEM, TSEST, description)
ANA_AGG_SERIES = [
    ("GDP",     "M1", "GPM",     "20", "GDP chain volume, seasonally adjusted (AUD millions)"),
    ("GDP_PCA", "M1", "GPM_PCA", "20", "GDP per capita, seasonally adjusted"),
    ("HSR",     "M7", "HSR",     "20", "Household saving ratio, seasonally adjusted (%)"),
    ("TOT",     "M5", "TTR",     "20", "Terms of trade index, seasonally adjusted"),
]


def load_config() -> dict:
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


# ─────────────────────────────────────────────────────────────────────────────
# SDMX period parser
# ─────────────────────────────────────────────────────────────────────────────

def parse_sdmx_period(period: str) -> pd.Timestamp:
    """Convert SDMX time period to Timestamp. Handles YYYY-QX and YYYY-MM."""
    period = period.strip()
    if "-Q" in period:
        year, q = period.split("-Q")
        month = (int(q) - 1) * 3 + 1
        return pd.Timestamp(f"{year}-{month:02d}-01")
    elif len(period) == 7 and period[4] == "-":
        return pd.Timestamp(f"{period}-01")
    return pd.Timestamp(period)


# ─────────────────────────────────────────────────────────────────────────────
# ANA_AGG XML parser
# ─────────────────────────────────────────────────────────────────────────────

def extract_from_xml(
    xml_path: Path,
    measure: str,
    data_item: str,
    tsest: str,
    canonical_name: str,
    description: str,
    start: str,
    out_dir: Path,
) -> dict:
    """Extract one series from the ANA_AGG SDMX XML file."""
    log.info(f"Extracting {canonical_name} (MEASURE={measure} DATA_ITEM={data_item} TSEST={tsest}) ...")

    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
    except Exception as exc:
        return error_result(canonical_name, f"XML parse error: {exc}")

    records = []
    for series in root.findall(".//generic:Series", NS):
        key = {
            v.get("id"): v.get("value")
            for v in series.findall("generic:SeriesKey/generic:Value", NS)
        }
        if (key.get("MEASURE") != measure or
                key.get("DATA_ITEM") != data_item or
                key.get("TSEST") != tsest):
            continue

        for obs in series.findall("generic:Obs", NS):
            period_el = obs.find("generic:ObsDimension", NS)
            value_el  = obs.find("generic:ObsValue", NS)
            if period_el is None:
                continue
            try:
                dt = parse_sdmx_period(period_el.get("value", ""))
            except Exception:
                continue
            val = float("nan")
            if value_el is not None and value_el.get("value"):
                try:
                    val = float(value_el.get("value"))
                except (ValueError, TypeError):
                    pass
            records.append({"date": dt, "value": val})
        break  # stop after first matching series

    if not records:
        return error_result(
            canonical_name,
            f"No series found matching MEASURE={measure} DATA_ITEM={data_item} TSEST={tsest}"
        )

    df = (pd.DataFrame(records)
            .assign(
                date=lambda d: pd.to_datetime(d["date"]),
                value=lambda d: pd.to_numeric(d["value"], errors="coerce"),
                series_id=f"{measure}_{data_item}_{tsest}",
                series_name=canonical_name,
                frequency="quarterly",
                source="ABS_ANA_AGG_XML",
            )
            .dropna(subset=["date"])
            .sort_values("date")
            .drop_duplicates("date")
            .reset_index(drop=True))

    return save(canonical_name, description, df, start, out_dir)


# ─────────────────────────────────────────────────────────────────────────────
# WPI Excel parser  (ABS cat 6345.0)
# ─────────────────────────────────────────────────────────────────────────────

def parse_wpi_xlsx(path: Path, start: str, out_dir: Path) -> dict:
    """
    Extract WPI: total hourly rates, all sectors (private + public),
    seasonally adjusted. Series ID A2713849C.
    Sheet: Data1, Series ID row has row[0] == 'Series ID'.
    """
    log.info(f"Extracting WPI from {path.name} ...")

    try:
        wb = openpyxl.load_workbook(path, read_only=True, data_only=True)
    except Exception as exc:
        return error_result("WPI", f"Cannot open xlsx: {exc}")

    if "Data1" not in wb.sheetnames:
        return error_result("WPI", f"Sheet 'Data1' not found. Sheets: {wb.sheetnames}")

    ws   = wb["Data1"]
    rows = list(ws.iter_rows(values_only=True))

    # Find Series ID row
    series_row_idx = next(
        (i for i, row in enumerate(rows) if row and row[0] == "Series ID"), None
    )
    if series_row_idx is None:
        return error_result("WPI", "No 'Series ID' row found in Data1 sheet")

    target    = "A2713849C"  # Private and Public combined, seasonally adjusted
    series_ids = list(rows[series_row_idx])
    if target not in series_ids:
        return error_result("WPI", f"Series {target} not found. Got: {[s for s in series_ids if s][:10]}")

    col_idx = series_ids.index(target)

    records = []
    for row in rows[series_row_idx + 1:]:
        if not row or row[0] is None:
            continue
        dt = row[0]
        if not isinstance(dt, datetime):
            try:
                dt = pd.to_datetime(dt)
            except Exception:
                continue
        val = row[col_idx]
        try:
            val = float(val) if val is not None else float("nan")
        except (TypeError, ValueError):
            val = float("nan")
        records.append({"date": pd.Timestamp(dt), "value": val})

    if not records:
        return error_result("WPI", "No data rows parsed from Data1 sheet")

    df = (pd.DataFrame(records)
            .assign(
                series_id="A2713849C",
                series_name="WPI",
                frequency="quarterly",
                source="ABS_6345_XLSX",
            )
            .dropna(subset=["date"])
            .sort_values("date")
            .drop_duplicates("date")
            .reset_index(drop=True))

    return save(
        "WPI",
        "Wage price index — total hourly rates all sectors SA (quarterly)",
        df, start, out_dir
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
    if len(df) < 40:
        issues.append({"code": "LOW_ROW_COUNT", "count": len(df)})

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
    parser.add_argument("--xml", default=None, help="Path to all.xml (ANA_AGG)")
    parser.add_argument("--wpi", default=None, help="Path to 634501.xlsx (WPI)")
    args = parser.parse_args()

    cfg   = load_config()
    start = cfg["data"]["start_date"]

    gdp_dir  = PROJECT_ROOT / cfg["raw"]["abs_gdp"]
    wpi_dir  = PROJECT_ROOT / cfg["raw"]["abs_wpi"]
    abs_base = PROJECT_ROOT / "data" / "raw" / "abs"

    log.info("=" * 60)
    log.info("fetch_abs.py — ABS manual file processor")
    log.info(f"Start date : {start}")
    log.info(f"Downloads  : {DOWNLOADS_DIR}")
    log.info("=" * 60)

    results = []

    # all.xml — GDP, GDP_PCA, HSR, TOT
    xml_path = resolve(args.xml, "xml")
    if xml_path:
        for canonical_name, measure, data_item, tsest, description in ANA_AGG_SERIES:
            result = extract_from_xml(
                xml_path, measure, data_item, tsest,
                canonical_name, description, start, gdp_dir
            )
            results.append(result)
    else:
        log.warning(
            "all.xml not found in downloads dir. "
            "Download from ABS Data Explorer (ANA_AGG dataflow) and place in "
            f"{DOWNLOADS_DIR}"
        )

    # 634501.xlsx — WPI
    wpi_path = resolve(args.wpi, "wpi")
    if wpi_path:
        results.append(parse_wpi_xlsx(wpi_path, start, wpi_dir))
    else:
        log.warning(
            "634501.xlsx not found in downloads dir. "
            "Download Table 1 from ABS Wage Price Index latest release and place in "
            f"{DOWNLOADS_DIR}"
        )

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

    abs_base.mkdir(parents=True, exist_ok=True)
    gdp_dir.mkdir(parents=True, exist_ok=True)
    wpi_dir.mkdir(parents=True, exist_ok=True)
    summary_path = abs_base / "_fetch_summary.json"
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

    log.info("ABS fetch complete.")


if __name__ == "__main__":
    main()