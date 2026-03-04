#!/usr/bin/env python3
"""
fetch_calendar.py
=================
Generates the ASX trading calendar from the AXJO index parquet file
and saves it to data/raw/calendar/.

The calendar is a core dependency for build_feature_panel.py. It defines
the universe of valid trading dates that all series are aligned to.

Why derive from AXJO (not pandas_market_calendars)?
  AXJO's actual trading dates are the ground truth for this pipeline.
  pandas_market_calendars approximates ASX holidays but can diverge
  from reality (ad-hoc closures, early closes). Since AXJO.parquet is
  already fetched and audited, using it as the calendar source ensures
  perfect alignment between equity prices and macro features.

What this produces:
  1. ASX_CALENDAR.parquet — full date range with columns:
       date, is_trading_day, day_of_week, week_of_year, month,
       quarter, year, is_month_end, is_quarter_end, is_year_end,
       days_to_next_trading, days_from_prev_trading

  2. TRADING_DATES.parquet — just the trading dates (simple lookup):
       date

  3. calendar_manifest.json — metadata and stats

Features generated (all boolean/integer, no floats):
  - cal_day_of_week:        0=Mon ... 4=Fri
  - cal_week_of_year:       1–53
  - cal_month:              1–12
  - cal_quarter:            1–4
  - cal_is_month_end:       1 if last trading day of month
  - cal_is_quarter_end:     1 if last trading day of quarter
  - cal_is_year_end:        1 if last trading day of year
  - cal_dist_to_month_end:  trading days to month end
  - cal_dist_to_year_end:   trading days to year end

Usage:
    python -m scripts.fetch.calendar.fetch_calendar

Output:
    data/raw/calendar/ASX_CALENDAR.parquet
    data/raw/calendar/TRADING_DATES.parquet
    data/raw/calendar/calendar_manifest.json
"""

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
        logging.FileHandler(PROJECT_ROOT / "logs" / "fetch_calendar.log"),
    ],
)
log = logging.getLogger(__name__)

CONFIG_PATH = PROJECT_ROOT / "config" / "data.yaml"


def load_config() -> dict:
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


def load_axjo(cfg: dict) -> pd.DataFrame:
    """Load AXJO.parquet — the source of ground-truth trading dates."""
    axjo_path = PROJECT_ROOT / cfg["raw"]["macro_indices"] / "AXJO.parquet"
    if not axjo_path.exists():
        raise FileNotFoundError(
            f"AXJO.parquet not found at {axjo_path}. "
            "Run fetch_macro_indices.py first."
        )
    df = pd.read_parquet(axjo_path, columns=["date"])
    df["date"] = pd.to_datetime(df["date"])
    df = df.dropna().sort_values("date").drop_duplicates("date")
    log.info(
        f"AXJO trading dates loaded: {len(df)} | "
        f"{df['date'].min().date()} → {df['date'].max().date()}"
    )
    return df


def build_calendar(trading_dates: pd.Series,
                   global_start: str,
                   global_end: str) -> pd.DataFrame:
    """
    Build a full calendar DataFrame covering global_start to global_end.
    Each row is one calendar day. Trading days are flagged.
    Calendar features are computed only for trading days.
    """
    # Full date range (every calendar day)
    all_dates = pd.date_range(
        start=pd.Timestamp(global_start),
        end=pd.Timestamp(global_end),
        freq="D"
    )

    trading_set = set(trading_dates.dt.normalize())

    df = pd.DataFrame({"date": all_dates})
    df["is_trading_day"] = df["date"].isin(trading_set).astype(int)

    # Basic calendar features (for all days — models use these as known futures)
    df["cal_day_of_week"]  = df["date"].dt.dayofweek      # 0=Mon, 4=Fri
    df["cal_week_of_year"] = df["date"].dt.isocalendar().week.astype(int)
    df["cal_month"]        = df["date"].dt.month
    df["cal_quarter"]      = df["date"].dt.quarter
    df["cal_year"]         = df["date"].dt.year

    # ── Trading-day-specific features ─────────────────────────────────────────
    # Work on trading days only, then merge back
    td = df[df["is_trading_day"] == 1][["date", "cal_month",
                                         "cal_quarter", "cal_year"]].copy()
    td = td.reset_index(drop=True)
    td["td_idx"] = td.index  # sequential trading day index

    # Last trading day of month / quarter / year
    td["cal_is_month_end"] = (
        td.groupby(["cal_year", "cal_month"])["td_idx"]
        .transform("max") == td["td_idx"]
    ).astype(int)

    td["cal_is_quarter_end"] = (
        td.groupby(["cal_year", "cal_quarter"])["td_idx"]
        .transform("max") == td["td_idx"]
    ).astype(int)

    td["cal_is_year_end"] = (
        td.groupby("cal_year")["td_idx"]
        .transform("max") == td["td_idx"]
    ).astype(int)

    # Distance to month/year end (in trading days)
    # For each trading day: how many trading days until the last of its month/year
    month_ends = td[td["cal_is_month_end"] == 1].set_index(["cal_year", "cal_month"])["td_idx"]
    td["month_end_td_idx"] = td.apply(
        lambda r: month_ends.get((r["cal_year"], r["cal_month"]), r["td_idx"]),
        axis=1
    )
    td["cal_dist_to_month_end"] = (td["month_end_td_idx"] - td["td_idx"]).clip(lower=0)

    year_ends = td[td["cal_is_year_end"] == 1].set_index("cal_year")["td_idx"]
    td["year_end_td_idx"] = td["cal_year"].map(year_ends).fillna(td["td_idx"])
    td["cal_dist_to_year_end"] = (td["year_end_td_idx"] - td["td_idx"]).clip(lower=0)

    # Merge calendar features back onto full date range
    cal_cols = [
        "date", "cal_is_month_end", "cal_is_quarter_end", "cal_is_year_end",
        "cal_dist_to_month_end", "cal_dist_to_year_end", "td_idx"
    ]
    df = df.merge(td[cal_cols], on="date", how="left")

    # td_idx is only meaningful for trading days — fill with -1 for non-trading
    df["td_idx"] = df["td_idx"].fillna(-1).astype(int)

    # Fill calendar features with 0 for non-trading days
    for col in ["cal_is_month_end", "cal_is_quarter_end", "cal_is_year_end",
                "cal_dist_to_month_end", "cal_dist_to_year_end"]:
        df[col] = df[col].fillna(0).astype(int)

    log.info(
        f"Calendar built: {len(df)} total days | "
        f"{df['is_trading_day'].sum()} trading days | "
        f"{(df['is_trading_day'] == 0).sum()} non-trading days"
    )
    return df


def validate(df: pd.DataFrame, trading_df: pd.DataFrame) -> list:
    """Basic validation checks."""
    issues = []

    n_trading = df["is_trading_day"].sum()
    if n_trading < 4000:
        issues.append({"code": "LOW_TRADING_DAY_COUNT", "count": int(n_trading)})

    # Check roughly 252 trading days per year (allow 240–265)
    by_year = df[df["is_trading_day"] == 1].groupby("cal_year").size()
    bad_years = by_year[(by_year < 240) | (by_year > 265)]
    if len(bad_years) > 1:  # Allow 1 partial year
        issues.append({
            "code":      "ABNORMAL_TRADING_DAY_COUNT",
            "years":     bad_years.to_dict(),
            "note":      "Expected 240–265 trading days per full year",
        })

    # Check no trading days on weekends
    td = df[df["is_trading_day"] == 1]
    weekend_trades = td[td["cal_day_of_week"] >= 5]
    if len(weekend_trades):
        issues.append({
            "code":  "TRADING_ON_WEEKEND",
            "count": len(weekend_trades),
            "dates": weekend_trades["date"].dt.strftime("%Y-%m-%d").tolist()[:5],
        })

    return issues


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    cfg = load_config()

    global_start = cfg["data"]["start_date"]
    from datetime import date
    global_end = cfg["data"]["end_date"] or date.today().isoformat()

    cal_dir = PROJECT_ROOT / cfg["raw"]["calendar"]
    cal_dir.mkdir(parents=True, exist_ok=True)

    log.info("=" * 60)
    log.info("fetch_calendar.py — ASX trading calendar")
    log.info(f"Date range : {global_start} → {global_end}")
    log.info(f"Source     : AXJO.parquet (ground truth trading dates)")
    log.info(f"Output     : {cal_dir}")
    log.info("=" * 60)

    # ── Load AXJO ─────────────────────────────────────────────────────────────
    axjo_df = load_axjo(cfg)

    # ── Build calendar ────────────────────────────────────────────────────────
    log.info("Building calendar features ...")
    cal_df = build_calendar(axjo_df["date"], global_start, global_end)

    # ── Validate ──────────────────────────────────────────────────────────────
    issues = validate(cal_df, axjo_df)
    status = "WARN" if issues else "OK"
    if issues:
        for iss in issues:
            log.warning(f"  {iss}")

    # ── Save ASX_CALENDAR.parquet ─────────────────────────────────────────────
    cal_path = cal_dir / "ASX_CALENDAR.parquet"
    cal_df.to_parquet(cal_path, index=False)
    log.info(f"Saved ASX_CALENDAR.parquet → {cal_path} ({len(cal_df)} rows)")

    # ── Save TRADING_DATES.parquet (simple lookup) ────────────────────────────
    trading_df = cal_df[cal_df["is_trading_day"] == 1][["date"]].copy()
    trading_path = cal_dir / "TRADING_DATES.parquet"
    trading_df.to_parquet(trading_path, index=False)
    log.info(f"Saved TRADING_DATES.parquet → {trading_path} ({len(trading_df)} rows)")

    # ── Stats by year ─────────────────────────────────────────────────────────
    td_by_year = (
        cal_df[cal_df["is_trading_day"] == 1]
        .groupby("cal_year").size()
        .reset_index()
        .rename(columns={0: "trading_days"})
    )
    log.info("Trading days per year:")
    for _, row in td_by_year.iterrows():
        log.info(f"  {int(row['cal_year'])}: {int(row['trading_days'])}")

    # ── Manifest ──────────────────────────────────────────────────────────────
    manifest = {
        "generated_at":    datetime.utcnow().isoformat(),
        "source":          "AXJO.parquet (ASX 200 index)",
        "global_start":    global_start,
        "global_end":      global_end,
        "status":          status,
        "total_days":      len(cal_df),
        "trading_days":    int(cal_df["is_trading_day"].sum()),
        "non_trading_days": int((cal_df["is_trading_day"] == 0).sum()),
        "date_min":        str(cal_df["date"].min().date()),
        "date_max":        str(cal_df["date"].max().date()),
        "trading_days_by_year": td_by_year.set_index("cal_year")["trading_days"].to_dict(),
        "issues":          issues,
        "outputs": {
            "ASX_CALENDAR":  str(cal_path),
            "TRADING_DATES": str(trading_path),
        },
        "columns": list(cal_df.columns),
    }

    manifest_path = cal_dir / "calendar_manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2, default=str)

    log.info("=" * 60)
    log.info("CALENDAR COMPLETE")
    log.info(f"  Status        : {status}")
    log.info(f"  Total days    : {manifest['total_days']}")
    log.info(f"  Trading days  : {manifest['trading_days']}")
    log.info(f"  date_min      : {manifest['date_min']}")
    log.info(f"  date_max      : {manifest['date_max']}")
    log.info(f"  Manifest      → {manifest_path}")
    log.info("=" * 60)
    log.info("Calendar ready. build_feature_panel.py can now align all series.")


if __name__ == "__main__":
    main()
