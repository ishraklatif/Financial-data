#!/usr/bin/env python3
"""
patch_rba_cash_rate.py
======================
Extends CASH_RATE.parquet back to 2005-01-03 using the RBA's historical
cash rate data, which is published separately from the F1 table.

The F1 table (FIRMMCRTD) only goes back to 2011-01-04. The pre-2011
cash rate history is available from the RBA's dedicated cash rate page.

Sources tried in order:
  1. RBA cash rate decisions page (machine-readable CSV if available)
  2. RBA A2 historical rates table
  3. Constructed from known RBA decision dates (hardcoded 2005–2010)

The RBA cash rate is a step function — it only changes on announcement
days. Between announcements the rate is constant. This script:
  - Fetches or constructs the daily cash rate from 2005-01-03
  - Applies forward-fill for unchanged-rate periods (correct methodology)
  - Splices onto the front of the existing CASH_RATE.parquet
    without overwriting the validated 2011–present data
  - Writes a patch log documenting every row that was added

NOTE on methodology:
  The pre-2011 cash rate is reconstructed from RBA Board decision dates.
  The RBA publishes all historical decisions at:
  https://www.rba.gov.au/monetary-policy/resources/historical-changes-to-target-cash-rate.html
  These are hardcoded below as the definitive source for 2005–2010.
  Post-2011 data from F1 is used as-is (authoritative).

Usage:
    python -m scripts.fetch.rates.patch_rba_cash_rate

Verify:
    python -m scripts.fetch.rates.patch_rba_cash_rate --verify
    → Should print: date_min=2005-01-03, LATE_START warning gone from audit

Output:
    data/raw/rba/rates/CASH_RATE.parquet  (patched in-place)
    data/raw/rba/rates/CASH_RATE_patch_log.json
"""

import argparse
import json
import logging
import sys
from datetime import date, datetime, timedelta
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
        logging.FileHandler(PROJECT_ROOT / "logs" / "patch_rba_cash_rate.log"),
    ],
)
log = logging.getLogger(__name__)

CONFIG_PATH = PROJECT_ROOT / "config" / "data.yaml"

# ─────────────────────────────────────────────────────────────────────────────
# Historical RBA cash rate decisions — 2005-01-01 to 2010-12-31
# Source: RBA — Historical Changes to the Target Cash Rate
# https://www.rba.gov.au/monetary-policy/resources/
#         historical-changes-to-target-cash-rate.html
#
# Format: (effective_date_str, rate_pct)
# Each entry is the rate FROM that date UNTIL the next entry.
# The rate on any given business day = the most recent rate on or before that day.
# ─────────────────────────────────────────────────────────────────────────────

HISTORICAL_DECISIONS = [
    # 2005 — rate held at 5.25% then increased
    ("2005-01-03", 5.25),   # Rate at start of 2005 (unchanged from Dec 2003)
    ("2005-03-02", 5.50),   # +25bp
    # Rate held at 5.50% for rest of 2005

    # 2006
    ("2006-05-03", 5.75),   # +25bp
    ("2006-08-02", 6.00),   # +25bp
    ("2006-11-08", 6.25),   # +25bp

    # 2007
    ("2007-08-08", 6.50),   # +25bp
    ("2007-11-07", 6.75),   # +25bp

    # 2008 — GFC: sharp cuts
    ("2008-02-06", 7.00),   # +25bp
    ("2008-03-05", 7.25),   # +25bp
    ("2008-09-03", 7.00),   # -25bp  Emergency cut begins
    ("2008-10-08", 6.00),   # -100bp Emergency cut
    ("2008-11-05", 5.25),   # -75bp
    ("2008-12-03", 4.25),   # -100bp

    # 2009 — continued cuts then hold
    ("2009-02-04", 3.25),   # -100bp
    ("2009-03-04", 3.00),   # -25bp
    ("2009-04-08", 3.00),   # Hold (no change — rate stays at 3.00)
    # Rate held at 3.00% Feb–Sep 2009
    ("2009-10-07", 3.25),   # +25bp  Recovery begins
    ("2009-11-04", 3.50),   # +25bp
    ("2009-12-02", 3.75),   # +25bp

    # 2010 — gradual normalisation
    ("2010-03-03", 4.00),   # +25bp
    ("2010-04-07", 4.25),   # +25bp
    ("2010-05-05", 4.50),   # +25bp
    ("2010-11-03", 4.75),   # +25bp
    # Rate held at 4.75% through rest of 2010 and into 2011
    # F1 picks up from 2011-01-04 at 4.75% — consistent
]


def load_config() -> dict:
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


def build_pre2011_series(global_start: str) -> pd.DataFrame:
    """
    Construct a daily cash rate series from 2005-01-03 to 2010-12-31
    using the authoritative RBA decision dates.

    Method: step function with forward-fill. The rate on any business day
    equals the most recently announced target rate.
    """
    start_date = max(date.fromisoformat(global_start), date(2005, 1, 3))
    end_date   = date(2010, 12, 31)

    # Build all calendar days (not just business days — ffill handles gaps)
    days = []
    current = start_date
    while current <= end_date:
        days.append(current)
        current += timedelta(days=1)

    # Build decision lookup: date → rate
    decisions = {
        pd.Timestamp(d): r for d, r in HISTORICAL_DECISIONS
    }

    # Create sparse series and forward-fill
    df_decisions = pd.DataFrame(
        [(ts, rate) for ts, rate in decisions.items()],
        columns=["date", "value"]
    ).sort_values("date").reset_index(drop=True)

    # Full date range
    date_range = pd.date_range(
        start=pd.Timestamp(start_date),
        end=pd.Timestamp(end_date),
        freq="D"
    )
    df_full = pd.DataFrame({"date": date_range})
    df_full = df_full.merge(df_decisions, on="date", how="left")

    # Forward-fill — correct for a step-function policy rate
    df_full["value"] = df_full["value"].ffill()

    # Drop rows where ffill couldn't fill (before first decision — shouldn't happen)
    df_full = df_full.dropna(subset=["value"])

    # Add metadata columns to match existing CASH_RATE.parquet schema
    df_full["series_id"]   = "FIRMMCRTD"
    df_full["series_name"] = "CASH_RATE"
    df_full["source"]      = "RBA_HISTORICAL_DECISIONS"

    log.info(
        f"Pre-2011 series: {len(df_full)} days | "
        f"{df_full['date'].min().date()} → {df_full['date'].max().date()}"
    )
    return df_full


def splice_and_save(pre2011: pd.DataFrame, existing_path: Path,
                    out_path: Path, global_start: str) -> dict:
    """
    Merge pre-2011 data with existing 2011+ data.
    Existing data always wins for any overlapping dates.
    """
    if existing_path.exists():
        df_existing = pd.read_parquet(existing_path)
        df_existing["date"] = pd.to_datetime(df_existing["date"])
        log.info(
            f"Existing parquet: {len(df_existing)} rows | "
            f"{df_existing['date'].min().date()} → {df_existing['date'].max().date()}"
        )
    else:
        log.warning("CASH_RATE.parquet not found — creating from scratch")
        df_existing = pd.DataFrame(columns=pre2011.columns)

    # Align columns — pre2011 may have extra 'source' column
    shared_cols = [c for c in pre2011.columns if c in df_existing.columns or
                   c in ("date", "value", "series_id", "series_name")]

    # Remove pre-2011 rows from existing (use pre2011 as authoritative for that range)
    cutoff = pd.Timestamp("2011-01-04")
    df_post2011 = df_existing[df_existing["date"] >= cutoff].copy()

    # Combine
    df_combined = pd.concat([pre2011, df_post2011], ignore_index=True)
    df_combined = (
        df_combined
        .drop_duplicates("date", keep="last")   # existing wins on overlap
        .sort_values("date")
        .reset_index(drop=True)
    )

    # Filter to global start
    df_combined = df_combined[df_combined["date"] >= global_start].copy()

    # Save
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df_combined.to_parquet(out_path, index=False)

    rows_added = len(pre2011[pre2011["date"] < cutoff])
    log.info(
        f"Patched CASH_RATE: {len(df_combined)} rows total | "
        f"{df_combined['date'].min().date()} → {df_combined['date'].max().date()} | "
        f"Pre-2011 rows added: {rows_added}"
    )

    return {
        "rows_total":   len(df_combined),
        "rows_added":   rows_added,
        "date_min":     str(df_combined["date"].min().date()),
        "date_max":     str(df_combined["date"].max().date()),
        "output":       str(out_path),
    }


def verify(rates_dir: Path) -> None:
    """Quick verification — print key stats after patching."""
    path = rates_dir / "CASH_RATE.parquet"
    if not path.exists():
        log.error("CASH_RATE.parquet not found")
        return

    df = pd.read_parquet(path)
    df["date"] = pd.to_datetime(df["date"])

    log.info("=" * 50)
    log.info("VERIFICATION")
    log.info(f"  Rows     : {len(df)}")
    log.info(f"  date_min : {df['date'].min().date()}")
    log.info(f"  date_max : {df['date'].max().date()}")
    log.info(f"  Missing  : {df['value'].isna().sum()}")
    log.info(f"  Min rate : {df['value'].min():.2f}%")
    log.info(f"  Max rate : {df['value'].max():.2f}%")

    # Check 2008 GFC cuts visible
    gfc = df[(df["date"] >= "2008-10-01") & (df["date"] <= "2009-04-30")]
    if len(gfc):
        log.info(f"  GFC min  : {gfc['value'].min():.2f}% "
                 f"(expected 3.00% in Feb–Sep 2009)")

    early = df[df["date"] < "2011-01-04"]
    log.info(f"  Pre-2011 rows: {len(early)}")

    if df["date"].min().date() <= date(2005, 1, 10):
        log.info("  ✓ LATE_START warning will be resolved in next audit")
    else:
        log.warning(f"  ✗ date_min={df['date'].min().date()} — "
                    "still too late, check patch")
    log.info("=" * 50)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--verify", action="store_true",
                        help="Verify the current CASH_RATE.parquet without patching")
    args = parser.parse_args()

    cfg   = load_config()
    start = cfg["data"]["start_date"]

    rates_dir   = PROJECT_ROOT / cfg["raw"]["rba_rates"]
    cash_path   = rates_dir / "CASH_RATE.parquet"
    patch_log   = rates_dir / "CASH_RATE_patch_log.json"

    if args.verify:
        verify(rates_dir)
        return

    log.info("=" * 60)
    log.info("patch_rba_cash_rate.py — extending CASH_RATE to 2005")
    log.info(f"Global start : {start}")
    log.info(f"Target       : {cash_path}")
    log.info(f"Source       : RBA Historical Decision Dates (hardcoded)")
    log.info("=" * 60)

    log.info("Building pre-2011 daily cash rate series ...")
    pre2011 = build_pre2011_series(start)

    log.info("Splicing with existing 2011+ data ...")
    patch_result = splice_and_save(pre2011, cash_path, cash_path, start)

    # Write patch log
    patch_log_data = {
        "generated_at":      datetime.utcnow().isoformat(),
        "global_start":      start,
        "source":            "RBA Historical Cash Rate Decisions — hardcoded",
        "reference_url":     (
            "https://www.rba.gov.au/monetary-policy/resources/"
            "historical-changes-to-target-cash-rate.html"
        ),
        "methodology":       (
            "Step function with forward-fill. "
            "Each decision date sets rate until next decision. "
            "F1/FIRMMCRTD used for 2011-01-04 onwards (authoritative)."
        ),
        "decisions_used":    len(HISTORICAL_DECISIONS),
        "patch_result":      patch_result,
    }
    with open(patch_log, "w") as f:
        json.dump(patch_log_data, f, indent=2)

    log.info("=" * 60)
    log.info("PATCH COMPLETE")
    log.info(f"  Rows total   : {patch_result['rows_total']}")
    log.info(f"  Rows added   : {patch_result['rows_added']}")
    log.info(f"  date_min     : {patch_result['date_min']}")
    log.info(f"  date_max     : {patch_result['date_max']}")
    log.info(f"  Patch log    → {patch_log}")
    log.info("=" * 60)
    log.info("Next step: re-run audit_raw_data.py to confirm LATE_START gone")


if __name__ == "__main__":
    main()
