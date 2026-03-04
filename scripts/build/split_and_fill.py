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
  Test  : 2023-01-01 → 2099-12-31   (open-ended, ~787+ trading days)

Rationale:
  - Train ends before COVID (2020-01-01) — model learns from normal
    and GFC regimes but is not contaminated by the COVID shock
  - Val covers COVID + recovery + rate-hike cycle (2020–2022) — most
    demanding out-of-sample period for regime generalization
  - Test covers post-hike, AI-boom, normalisation (2023–present) —
    true held-out period, never touched during model selection

WHY SPLIT-AWARE FFILL WITH BOUNDARY SEEDING
--------------------------------------------
Forward-filling AFTER splitting prevents a subtle leakage.
Additionally, a boundary-seeding step carries the last known value
from train's tail into val, and from val's tail into test, before
ffill runs within each split. This ensures that quarterly/monthly
series (e.g. GDP_AUS) do not start val/test with NaN simply because
the most recent release sits in the prior split.

COLUMN ORDER FIX
----------------
On save, all split parquets are reordered to match the locked schema
column order: [date, ticker, y_ret_21d, y_dir_21d, <features sorted>].
This ensures assert_schema.py --strict passes the column order check
without warnings.

FFILL COLUMNS
-------------
All macro series (both optimistic and _lag variants) are forward-filled:
  CPI_*, GDP_*, HSR_*, TOT_*, WPI_*, FEDFUNDS_*, PCEPI_*, INDPRO_*,
  PMI_*, JOLTS_*, UMICH_*, UNEMP_*, F3_A_*, F3_BBB_*

Short interest (si_*) is also forward-filled: biweekly reporting.
Daily series (eq_*, xa_*, xs_*, yc_*, cr_*, cal_*, sr_*) are NOT
forward-filled — a missing daily value is a genuine data gap.

OUTPUTS
-------
data/features/panel_train.parquet
data/features/panel_val.parquet
data/features/panel_test.parquet
data/features/splits_manifest.json  — boundary dates, row counts, col counts

USAGE
-----
    python -m scripts.build.split_and_fill
    python -m scripts.build.split_and_fill --verify
    python -m scripts.build.split_and_fill --dry-run   # print stats, don't save
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
    "test":  ("2023-01-01", "2099-12-31"),   # open-ended upper bound
}

# ─────────────────────────────────────────────────────────────────────────────
# Forward-fill column groups
# ─────────────────────────────────────────────────────────────────────────────

# Prefixes of columns that should be forward-filled within each split.
# All other columns are left as-is (NaN = genuine gap).
FFILL_PREFIXES = (
    # AUS macro (quarterly / monthly, both optimistic and _lag)
    "CPI_INDEX", "CPI_YOY",
    "UNEMP_AUS",
    "GDP_AUS", "GDP_PCA_AUS",
    "HSR",
    "TOT",
    "WPI",
    # US macro (monthly, both optimistic and _lag)
    "CPI_US", "PCEPI",
    "UNEMP_US",
    "FEDFUNDS",
    "GDP_US",
    "INDPRO",
    "PMI_MFG", "PMI_SERV",
    "JOLTS",
    "UMICH",
    # RBA F3 credit yields (monthly)
    "F3_A_YIELD", "F3_BBB_YIELD",
    # Short interest (biweekly)
    "si_",
)


def is_ffill_column(col: str) -> bool:
    return any(col.startswith(pfx) for pfx in FFILL_PREFIXES)


# ─────────────────────────────────────────────────────────────────────────────
# Core logic
# ─────────────────────────────────────────────────────────────────────────────

def split_panel(panel: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Slice the panel into train/val/test by date."""
    splits = {}
    for name, (start, end) in SPLITS.items():
        mask = (panel["date"] >= start) & (panel["date"] <= end)
        splits[name] = panel[mask].copy()
        log.info(f"  {name}: {start} → {end}  | "
                 f"rows={len(splits[name]):,}  "
                 f"tickers={splits[name]['ticker'].nunique()}")
    return splits


def apply_split_ffill(
    df: pd.DataFrame,
    split_name: str,
    seed_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    Forward-fill macro and short-interest columns within a split,
    independently per ticker, ordered by date.

    If seed_df is provided (the previous split), the last known value
    for each ffill column is injected as a seed row before filling,
    then removed. This ensures quarterly series don't start the split
    with NaN simply because the most recent release was in the prior split.
    """
    ffill_cols = [c for c in df.columns if is_ffill_column(c)]

    if not ffill_cols:
        log.warning(f"  [{split_name}] No ffill columns found — check FFILL_PREFIXES")
        return df

    log.info(f"  [{split_name}] Applying ffill to {len(ffill_cols)} columns "
             f"across {df['ticker'].nunique()} tickers ...")

    # Sort by ticker then date to ensure correct fill direction
    df = df.sort_values(["ticker", "date"]).reset_index(drop=True)

    # ── Boundary seeding ──────────────────────────────────────────────────
    if seed_df is not None:
        log.info(f"  [{split_name}] Seeding from previous split ...")
        seed_df = seed_df.sort_values(["ticker", "date"])

        # For each ticker, get the last row of the seed split for ffill cols
        seed_last = (
            seed_df.groupby("ticker", observed=True)[ffill_cols]
            .last()
            .reset_index()
        )

        # Build seed rows: one per ticker, dated 1 day before split start
        split_start = pd.Timestamp(SPLITS[split_name][0])
        seed_date   = split_start - pd.Timedelta(days=1)

        seed_rows = seed_last.copy()
        seed_rows["date"]      = seed_date
        seed_rows["y_ret_21d"] = np.nan
        seed_rows["y_dir_21d"] = np.nan
        seed_rows["_is_seed"]  = True

        # Fill any remaining non-ffill cols with NaN
        for col in df.columns:
            if col not in seed_rows.columns:
                seed_rows[col] = np.nan

        seed_rows = seed_rows[df.columns.tolist() + ["_is_seed"]]
        df["_is_seed"] = False

        # Combine seed rows + real split rows
        combined = pd.concat([seed_rows, df], ignore_index=True)
        combined = combined.sort_values(["ticker", "date"]).reset_index(drop=True)

        # Apply ffill per ticker across combined
        combined[ffill_cols] = (
            combined.groupby("ticker", observed=True)[ffill_cols]
            .transform(lambda x: x.ffill())
        )

        # Count cells seeded
        n_seeded = (
            combined[combined["_is_seed"] == True][ffill_cols]
            .notna().sum().sum()
        )
        log.info(f"  [{split_name}] Seeded {n_seeded:,} cells from previous split")

        # Drop seed rows
        df = combined[combined["_is_seed"] == False].drop(columns=["_is_seed"])
        df = df.reset_index(drop=True)

        # Restore integer dtypes promoted to float64 by NaN concat
        df = restore_dtypes(df, seed_df)

    else:
        # No seeding: just ffill within split per ticker
        df[ffill_cols] = (
            df.groupby("ticker", observed=True)[ffill_cols]
            .transform(lambda x: x.ffill())
        )

    # Verify: count remaining NaN in ffill cols
    remaining_null = df[ffill_cols].isna().mean().mean()
    log.info(f"  [{split_name}] Remaining null in ffill cols: "
             f"{remaining_null:.1%} "
             f"(expected: ~0% for long-history series, "
             f">0% for short-history tickers at start of their data)")

    return df


def validate_no_leakage(splits: dict[str, pd.DataFrame]) -> None:
    """Assert that no date appears in more than one split."""
    train_dates = set(splits["train"]["date"].dt.date)
    val_dates   = set(splits["val"]["date"].dt.date)
    test_dates  = set(splits["test"]["date"].dt.date)

    tv_overlap = train_dates & val_dates
    vt_overlap = val_dates   & test_dates
    tt_overlap = train_dates & test_dates

    if tv_overlap or vt_overlap or tt_overlap:
        log.error("DATE LEAKAGE DETECTED:")
        if tv_overlap:
            log.error(f"  Train ∩ Val: {sorted(tv_overlap)[:5]}")
        if vt_overlap:
            log.error(f"  Val ∩ Test: {sorted(vt_overlap)[:5]}")
        if tt_overlap:
            log.error(f"  Train ∩ Test: {sorted(tt_overlap)[:5]}")
        sys.exit(1)

    log.info("  No date overlap between splits (no leakage)")


def validate_column_consistency(splits: dict[str, pd.DataFrame]) -> None:
    """Assert all three splits have identical columns in the same order."""
    cols = {name: list(df.columns) for name, df in splits.items()}
    if cols["train"] != cols["val"] or cols["train"] != cols["test"]:
        log.error("Column mismatch between splits!")
        train_set = set(cols["train"])
        val_set   = set(cols["val"])
        test_set  = set(cols["test"])
        log.error(f"  In train not val: {train_set - val_set}")
        log.error(f"  In val not train: {val_set - train_set}")
        log.error(f"  In train not test: {train_set - test_set}")
        sys.exit(1)
    log.info(f"  All splits have {len(cols['train'])} columns in identical order")


def restore_dtypes(df: pd.DataFrame, ref_df: pd.DataFrame) -> pd.DataFrame:
    """
    Restore integer dtypes that were promoted to float64 by the boundary-seeding
    step (pandas upcasts int columns when NaN seed rows are concatenated).

    Only restores columns where:
      - ref_df dtype is integer (int32, int64, etc.)
      - the column has no NaN values in df (safe to downcast)

    This is applied to val and test after seed rows are removed.
    """
    for col in df.columns:
        if col not in ref_df.columns:
            continue
        ref_dtype = ref_df[col].dtype
        cur_dtype = df[col].dtype
        if ref_dtype == cur_dtype:
            continue
        # Only attempt restore for integer targets
        if not pd.api.types.is_integer_dtype(ref_dtype):
            continue
        # Only safe if no NaN remaining
        if df[col].isna().any():
            continue
        try:
            df[col] = df[col].astype(ref_dtype)
        except (ValueError, OverflowError):
            pass  # leave as-is if conversion fails
    return df


def reorder_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Reorder columns to canonical order:
      [date, ticker, y_ret_21d, y_dir_21d, <all feature cols alphabetically>]

    This ensures assert_schema --strict passes without column-order warnings.
    The locked schema in features_locked.json stores features sorted
    alphabetically, so this must match.
    """
    meta_cols    = ["date", "ticker", "y_ret_21d", "y_dir_21d"]
    feature_cols = sorted([c for c in df.columns if c not in set(meta_cols)])
    return df[meta_cols + feature_cols]


def build_manifest(
    splits: dict[str, pd.DataFrame],
    ffill_cols: list[str],
) -> dict:
    """Build splits_manifest.json."""
    manifest = {
        "generated_at": datetime.utcnow().isoformat(),
        "input":        str(INPUT_PATH),
        "ffill_prefixes": list(FFILL_PREFIXES),
        "ffill_col_count": len(ffill_cols),
        "splits": {},
    }

    for name, df in splits.items():
        ret = df["y_ret_21d"].dropna()
        manifest["splits"][name] = {
            "start":         str(df["date"].min().date()),
            "end":           str(df["date"].max().date()),
            "rows":          int(len(df)),
            "tickers":       int(df["ticker"].nunique()),
            "feature_cols":  int(len(df.columns) - 4),  # excl date, ticker, y_ret_21d, y_dir_21d
            "target_non_null": int(ret.notna().sum()),
            "target_pct_up": round(float((ret > 0).mean() * 100), 2),
            "target_mean":   round(float(ret.mean()), 6),
            "target_std":    round(float(ret.std()), 6),
            "output":        str(OUTPUT_PATHS[name]),
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

    # ── Load ──────────────────────────────────────────────────────────────
    log.info("Loading panel_with_targets.parquet ...")
    panel = pd.read_parquet(INPUT_PATH)
    panel["date"] = pd.to_datetime(panel["date"])
    log.info(f"  Loaded: {len(panel):,} rows | {panel['ticker'].nunique()} tickers | "
             f"{len(panel.columns)} columns")

    # ── Identify ffill columns ────────────────────────────────────────────
    ffill_cols = [c for c in panel.columns if is_ffill_column(c)]
    log.info(f"  ffill columns: {len(ffill_cols)} — sample: {ffill_cols[:4]}")

    # ── Split ─────────────────────────────────────────────────────────────
    log.info("Splitting ...")
    splits = split_panel(panel)

    # ── Validate split integrity ──────────────────────────────────────────
    log.info("Validating ...")
    validate_no_leakage(splits)
    validate_column_consistency(splits)

    # ── Apply split-aware ffill with boundary seeding ─────────────────────
    log.info("Applying split-aware forward fill with boundary seeding ...")
    splits["train"] = apply_split_ffill(splits["train"], "train", seed_df=None)
    splits["val"]   = apply_split_ffill(splits["val"],   "val",   seed_df=splits["train"])
    splits["test"]  = apply_split_ffill(splits["test"],  "test",  seed_df=splits["val"])

    # ── Reorder columns to match locked schema (alphabetical features) ────
    log.info("Reordering columns to canonical schema order ...")
    for name in ["train", "val", "test"]:
        splits[name] = reorder_columns(splits[name])
    log.info(f"  Column order: date, ticker, y_ret_21d, y_dir_21d, "
             f"<{len(splits['train'].columns) - 4} features alphabetically>")

    # ── Print summary ─────────────────────────────────────────────────────
    log.info("=" * 65)
    log.info("SPLIT SUMMARY (after ffill):")
    for name, df in splits.items():
        ret = df["y_ret_21d"].dropna()
        null_after = df[ffill_cols].isna().mean().mean()
        log.info(f"  {name:<6}  rows={len(df):>7,}  "
                 f"tickers={df['ticker'].nunique():>2}  "
                 f"target={len(ret):>7,} ({len(ret)/len(df):.1%})  "
                 f"pct_up={float((ret>0).mean()*100):.1f}%  "
                 f"macro_null={null_after:.1%}")

    # ── Spot-check GDP ────────────────────────────────────────────────────
    sample_col = next((c for c in ffill_cols if c.startswith("GDP_AUS") and "_lag" not in c), None)
    if sample_col:
        log.info(f"Spot-check '{sample_col}' null rate by split:")
        for name, df in splits.items():
            null_pct = df[sample_col].isna().mean()
            log.info(f"  [{name}] {null_pct:.2%}  "
                     f"(expected <1% for full-history tickers)")

    if dry_run:
        log.info("DRY RUN — not saving. Pass without --dry-run to write files.")
        return

    # ── Save ──────────────────────────────────────────────────────────────
    log.info("Saving ...")
    for name, df in splits.items():
        path = OUTPUT_PATHS[name]
        df.to_parquet(path, index=False, engine="pyarrow", compression="snappy")
        size_mb = path.stat().st_size / 1e6
        log.info(f"  {name}: {path.name}  ({size_mb:.1f} MB)")

    # ── Manifest ──────────────────────────────────────────────────────────
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
                 f"date=[{df['date'].min().date()} → {df['date'].max().date()}]  "
                 f"target={len(ret):,} ({len(ret)/len(df):.1%})  "
                 f"macro_null={macro_null:.1%}")

        start, end = SPLITS[name]
        if df["date"].min() < pd.Timestamp(start):
            log.warning(f"  [{name}] date min {df['date'].min().date()} < expected {start}")
        if name != "test" and df["date"].max() > pd.Timestamp(end):
            log.warning(f"  [{name}] date max {df['date'].max().date()} > expected {end}")

    if MANIFEST_PATH.exists():
        with open(MANIFEST_PATH) as f:
            manifest = json.load(f)
        log.info(f"  Manifest generated_at: {manifest['generated_at']}")
        log.info(f"  ffill columns: {manifest['ffill_col_count']}")

    if all_ok:
        log.info("  ✓ All splits present and valid")
    else:
        log.error("  ✗ Some splits missing — re-run without --verify")
        sys.exit(1)

    log.info("=" * 65)


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--verify",  action="store_true",
                        help="Verify existing split parquets without rerunning")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print statistics without writing files")
    args = parser.parse_args()

    if args.verify:
        verify()
    elif args.dry_run:
        run(dry_run=True)
    else:
        run(dry_run=False)