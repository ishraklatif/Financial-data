#!/usr/bin/env python3
"""
split_and_fill.py
=================
Phase 3, Step 3 — Temporal split and split-aware forward fill.

Reads panel_with_targets.parquet and produces three non-overlapping
splits on the date axis, then applies forward-fill to low-frequency
macro series WITHIN each split independently.

SPLIT BOUNDARIES
----------------
  Train : 2005-01-04 → 2019-12-31   (~15 years, ~3800 trading days)
  Val   : 2020-01-01 → 2022-12-31   (~3 years,  ~756 trading days)
  Test  : 2023-01-01 → 2026-02-26   (~3 years,  ~787 trading days)

Rationale:
  - Train ends before COVID (2020-01-01) — model learns from normal
    and GFC regimes but is not contaminated by the COVID shock
  - Val covers COVID + recovery + rate-hike cycle (2020-2022) — most
    demanding out-of-sample period for regime generalization
  - Test covers post-hike, AI-boom, normalisation (2023-present) —
    true held-out period, never touched during model selection

WHY SPLIT-AWARE FFILL
----------------------
Forward-filling AFTER splitting prevents a subtle leakage:
  If you ffill globally then split, the last macro value in train
  propagates into val as if it were val's starting value. That is
  correct and not leakage. BUT the scaler fitted on train would have
  seen a different distribution for that macro series than if it had
  been naturally sparse at the split boundary. Split-then-fill is the
  conservative and correct approach adopted by production ML pipelines.

FFILL COLUMNS
-------------
All macro series (both optimistic and _lag variants) are forward-filled.
Short interest (si_*) is also forward-filled: biweekly reporting.
Daily series (eq_*, xa_*, xs_*, yc_*, cr_*, cal_*, sr_*) are NOT ffilled.

OUTPUTS
-------
data/features/panel_train.parquet
data/features/panel_val.parquet
data/features/panel_test.parquet
data/features/splits_manifest.json

USAGE
-----
    python -m scripts.build.split_and_fill
    python -m scripts.build.split_and_fill --verify
    python -m scripts.build.split_and_fill --dry-run
"""

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(PROJECT_ROOT / "logs" / "split_and_fill.log"),
    ],
)
log = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────────────────────────────────────

FEATURES_DIR = PROJECT_ROOT / "data" / "features"
INPUT_PATH   = FEATURES_DIR / "panel_with_targets.parquet"

OUTPUT_PATHS = {
    "train": FEATURES_DIR / "panel_train.parquet",
    "val":   FEATURES_DIR / "panel_val.parquet",
    "test":  FEATURES_DIR / "panel_test.parquet",
}
MANIFEST_PATH = FEATURES_DIR / "splits_manifest.json"

# ─────────────────────────────────────────────────────────────────────────────
# Split boundaries
# ─────────────────────────────────────────────────────────────────────────────

SPLITS = {
    "train": ("2005-01-04", "2019-12-31"),
    "val":   ("2020-01-01", "2022-12-31"),
    "test":  ("2023-01-01", "2099-12-31"),
}

# ─────────────────────────────────────────────────────────────────────────────
# Forward-fill column groups
# ─────────────────────────────────────────────────────────────────────────────

FFILL_PREFIXES = (
    "CPI_INDEX", "CPI_YOY",
    "UNEMP_AUS",
    "GDP_AUS", "GDP_PCA_AUS",
    "HSR",
    "TOT",
    "WPI",
    "CPI_US", "PCEPI",
    "UNEMP_US",
    "FEDFUNDS",
    "GDP_US",
    "INDPRO",
    "PMI_MFG", "PMI_SERV",
    "JOLTS",
    "UMICH",
    "F3_A_YIELD", "F3_BBB_YIELD",
    "si_",
)


def is_ffill_column(col: str) -> bool:
    return any(col.startswith(pfx) for pfx in FFILL_PREFIXES)


# ─────────────────────────────────────────────────────────────────────────────
# Core logic
# ─────────────────────────────────────────────────────────────────────────────

def split_panel(panel: pd.DataFrame) -> dict:
    splits = {}
    for name, (start, end) in SPLITS.items():
        mask = (panel["date"] >= start) & (panel["date"] <= end)
        splits[name] = panel[mask].copy()
        log.info(f"  {name}: {start} -> {end}  |  "
                 f"rows={len(splits[name]):,}  "
                 f"tickers={splits[name]['ticker'].nunique()}")
    return splits


def get_split_seeds(prev_df: pd.DataFrame, ffill_cols: list) -> pd.DataFrame:
    """
    Extract the last known value of each ffill column per ticker from
    the previous split. Used to seed the next split so macro values
    don't go null at split boundaries.
    Returns a DataFrame indexed by ticker with one row per ticker.
    """
    prev_sorted = prev_df.sort_values(["ticker", "date"])
    seeds = (
        prev_sorted.groupby("ticker", observed=True)[ffill_cols]
        .last()  # last() returns last non-null value per group
    )
    return seeds


def apply_split_ffill(
    df: pd.DataFrame,
    split_name: str,
    seed_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    Forward-fill macro and short-interest columns within a split,
    independently per ticker, ordered by date.

    seed_df: optional DataFrame (indexed by ticker) with last known
             values from the previous split. Fills NaN at the start
             of this split before applying ffill.
    """
    ffill_cols = [c for c in df.columns if is_ffill_column(c)]

    if not ffill_cols:
        log.warning(f"  [{split_name}] No ffill columns found")
        return df

    log.info(f"  [{split_name}] Applying ffill to {len(ffill_cols)} columns "
             f"across {df['ticker'].nunique()} tickers ...")

    df = df.sort_values(["ticker", "date"]).reset_index(drop=True)

    # Seed: inject last known values from previous split into the
    # first row of each ticker in this split, where values are NaN.
    # This prevents a quarter-long null gap at split boundaries.
    if seed_df is not None:
        log.info(f"  [{split_name}] Seeding from previous split ...")
        seeded = 0
        for ticker in df["ticker"].unique():
            if ticker not in seed_df.index:
                continue
            ticker_mask = df["ticker"] == ticker
            ticker_idx  = df.index[ticker_mask]
            if len(ticker_idx) == 0:
                continue
            first_idx = ticker_idx[0]
            for col in ffill_cols:
                if col not in seed_df.columns:
                    continue
                seed_val = seed_df.loc[ticker, col]
                if pd.isna(seed_val):
                    continue
                # Only seed if the first row of this ticker is NaN
                if pd.isna(df.loc[first_idx, col]):
                    df.loc[first_idx, col] = seed_val
                    seeded += 1
        log.info(f"  [{split_name}] Seeded {seeded:,} cells from previous split")

    # Apply ffill within split per ticker
    df[ffill_cols] = (
        df.groupby("ticker", observed=True)[ffill_cols]
          .transform(lambda x: x.ffill())
    )

    remaining_null = df[ffill_cols].isna().mean().mean()
    log.info(f"  [{split_name}] Remaining null in ffill cols: {remaining_null:.1%}")
    return df


def validate_no_leakage(splits: dict) -> None:
    train_dates = set(splits["train"]["date"].dt.date)
    val_dates   = set(splits["val"]["date"].dt.date)
    test_dates  = set(splits["test"]["date"].dt.date)

    tv = train_dates & val_dates
    vt = val_dates   & test_dates
    tt = train_dates & test_dates

    if tv or vt or tt:
        log.error("DATE LEAKAGE DETECTED:")
        if tv: log.error(f"  Train & Val overlap: {sorted(tv)[:5]}")
        if vt: log.error(f"  Val & Test overlap:  {sorted(vt)[:5]}")
        if tt: log.error(f"  Train & Test overlap: {sorted(tt)[:5]}")
        sys.exit(1)
    log.info("  No date overlap between splits (no leakage)")


def validate_column_consistency(splits: dict) -> None:
    cols = {name: list(df.columns) for name, df in splits.items()}
    if cols["train"] != cols["val"] or cols["train"] != cols["test"]:
        log.error("Column mismatch between splits!")
        sys.exit(1)
    log.info(f"  All splits have {len(cols['train'])} columns in identical order")


def build_manifest(splits: dict, ffill_cols: list) -> dict:
    manifest = {
        "generated_at":    datetime.utcnow().isoformat(),
        "input":           str(INPUT_PATH),
        "ffill_prefixes":  list(FFILL_PREFIXES),
        "ffill_col_count": len(ffill_cols),
        "splits": {},
    }
    for name, df in splits.items():
        ret = df["y_ret_21d"].dropna()
        manifest["splits"][name] = {
            "start":           str(df["date"].min().date()),
            "end":             str(df["date"].max().date()),
            "rows":            int(len(df)),
            "tickers":         int(df["ticker"].nunique()),
            "feature_cols":    int(len(df.columns) - 4),
            "target_non_null": int(len(ret)),
            "target_pct_up":   round(float((ret > 0).mean() * 100), 2),
            "target_mean":     round(float(ret.mean()), 6),
            "target_std":      round(float(ret.std()), 6),
            "output":          str(OUTPUT_PATHS[name]),
        }
    return manifest


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def run(dry_run: bool = False) -> None:
    log.info("=" * 65)
    log.info("split_and_fill.py  Phase 3 Step 3")
    log.info(f"Input  : {INPUT_PATH}")
    log.info("=" * 65)

    if not INPUT_PATH.exists():
        log.error(f"panel_with_targets.parquet not found at {INPUT_PATH}")
        log.error("Run compute_targets.py first.")
        sys.exit(1)

    log.info("Loading panel_with_targets.parquet ...")
    panel = pd.read_parquet(INPUT_PATH)
    panel["date"] = pd.to_datetime(panel["date"])
    log.info(f"  Loaded: {len(panel):,} rows | "
             f"{panel['ticker'].nunique()} tickers | "
             f"{len(panel.columns)} columns")

    ffill_cols = [c for c in panel.columns if is_ffill_column(c)]
    log.info(f"  ffill columns: {len(ffill_cols)} — sample: {ffill_cols[:4]}")

    log.info("Splitting ...")
    splits = split_panel(panel)

    log.info("Validating ...")
    validate_no_leakage(splits)
    validate_column_consistency(splits)

    log.info("Applying split-aware forward fill with boundary seeding ...")
    ffill_cols = [c for c in panel.columns if is_ffill_column(c)]

    # Train: no seed (it is the first split)
    splits["train"] = apply_split_ffill(splits["train"], "train", seed_df=None)

    # Val: seed from last known train values per ticker
    train_seeds = get_split_seeds(splits["train"], ffill_cols)
    splits["val"] = apply_split_ffill(splits["val"], "val", seed_df=train_seeds)

    # Test: seed from last known val values per ticker
    val_seeds = get_split_seeds(splits["val"], ffill_cols)
    splits["test"] = apply_split_ffill(splits["test"], "test", seed_df=val_seeds)

    log.info("=" * 65)
    log.info("SPLIT SUMMARY (after ffill):")
    for name, df in splits.items():
        ret = df["y_ret_21d"].dropna()
        macro_null = df[ffill_cols].isna().mean().mean()
        log.info(f"  {name:<6}  rows={len(df):>7,}  "
                 f"tickers={df['ticker'].nunique():>2}  "
                 f"target={len(ret):>7,} ({len(ret)/len(df):.1%})  "
                 f"pct_up={float((ret>0).mean()*100):.1f}%  "
                 f"macro_null={macro_null:.1%}")

    # Spot-check: GDP null rate should be near 0 after ffill
    sample_macro = next(
        (c for c in ffill_cols if c.startswith("GDP_AUS") and "_lag" not in c), None
    )
    if sample_macro:
        log.info(f"Spot-check '{sample_macro}' null rate by split:")
        for name, df in splits.items():
            null_pct = df[sample_macro].isna().mean()
            log.info(f"  [{name}] {null_pct:.2%}  "
                     f"(expected <1% for full-history tickers)")

    if dry_run:
        log.info("DRY RUN complete — no files written.")
        return

    log.info("Saving ...")
    for name, df in splits.items():
        path = OUTPUT_PATHS[name]
        df.to_parquet(path, index=False, engine="pyarrow", compression="snappy")
        size_mb = path.stat().st_size / 1e6
        log.info(f"  {name}: {path.name}  ({size_mb:.1f} MB)")

    manifest = build_manifest(splits, ffill_cols)
    with open(MANIFEST_PATH, "w") as f:
        json.dump(manifest, f, indent=2)
    log.info(f"  Manifest: {MANIFEST_PATH}")

    log.info("=" * 65)
    log.info("SPLIT AND FILL COMPLETE")
    log.info(f"  Train  : {OUTPUT_PATHS['train']}")
    log.info(f"  Val    : {OUTPUT_PATHS['val']}")
    log.info(f"  Test   : {OUTPUT_PATHS['test']}")
    log.info("=" * 65)


def verify() -> None:
    log.info("=" * 65)
    log.info("VERIFICATION — split parquets")
    all_ok = True

    for name, path in OUTPUT_PATHS.items():
        if not path.exists():
            log.error(f"  {name}: NOT FOUND at {path}")
            all_ok = False
            continue

        df = pd.read_parquet(path)
        df["date"] = pd.to_datetime(df["date"])
        ret = df["y_ret_21d"].dropna()
        ffill_cols = [c for c in df.columns if is_ffill_column(c)]
        macro_null = df[ffill_cols].isna().mean().mean()

        log.info(f"  {name:<6}  rows={len(df):>7,}  "
                 f"date=[{df['date'].min().date()} to {df['date'].max().date()}]  "
                 f"target={len(ret):,} ({len(ret)/len(df):.1%})  "
                 f"macro_null={macro_null:.1%}")

    if MANIFEST_PATH.exists():
        with open(MANIFEST_PATH) as f:
            m = json.load(f)
        log.info(f"  Manifest: generated_at={m['generated_at']}  "
                 f"ffill_cols={m['ffill_col_count']}")

    if all_ok:
        log.info("  All splits present and valid")
    else:
        log.error("  Some splits missing — re-run without --verify")
        sys.exit(1)

    log.info("=" * 65)


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--verify",  action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if args.verify:
        verify()
    elif args.dry_run:
        run(dry_run=True)
    else:
        run(dry_run=False)