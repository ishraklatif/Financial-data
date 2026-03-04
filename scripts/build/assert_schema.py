#!/usr/bin/env python3
"""
assert_schema.py
================
Phase 3, Step 5 — Schema assertion for val and test splits.

Reads config/features_locked.json (written by lock_features.py) and
asserts that every split (train, val, test) is consistent with the
locked schema. This is the gate that must pass before any model
training or inference is attempted.

CHECKS PERFORMED
----------------
For each split:

  1. COLUMN PRESENCE   — every locked feature exists in the split
  2. COLUMN ORDER      — locked features appear in the same relative
                         order (extra non-feature cols are ignored)
  3. DTYPE CONSISTENCY — each locked feature has the same dtype as
                         in train (float32/64, int, bool)
  4. HASH VERIFICATION — the features_locked.json schema hash matches
                         the feature list on disk (detects tampering)
  5. NULL BUDGET       — each feature's null rate in val/test does not
                         exceed ALLOWED_NULL_MULTIPLIER × train null rate
                         (warns but does not fail — some extra nulls in
                         val/test are expected for post-2019 series)
  6. TARGET PRESENCE   — y_ret_21d and y_dir_21d are present in every
                         split
  7. DATE MONOTONICITY — dates are strictly increasing within each
                         ticker, no duplicate (date, ticker) pairs
  8. NO SPLIT LEAKAGE  — train end < val start < test start

EXIT CODES
----------
  0  all assertions passed (or only warnings)
  1  one or more hard failures

USAGE
-----
    python -m scripts.build.assert_schema              # check all splits
    python -m scripts.build.assert_schema --split val  # check one split
    python -m scripts.build.assert_schema --strict     # fail on warnings too
"""

import argparse
import hashlib
import json
import logging
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(PROJECT_ROOT / "logs" / "assert_schema.log"),
    ],
)
log = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────────────────────────────────────

FEATURES_DIR = PROJECT_ROOT / "data" / "features"
CONFIG_DIR   = PROJECT_ROOT / "config"
AUDIT_DIR    = PROJECT_ROOT / "data" / "audit"

LOCKED_PATH  = CONFIG_DIR / "features_locked.json"

SPLIT_PATHS = {
    "train": FEATURES_DIR / "panel_train.parquet",
    "val":   FEATURES_DIR / "panel_val.parquet",
    "test":  FEATURES_DIR / "panel_test.parquet",
}

# Null rate in val/test may be up to this multiple of train null rate before
# a WARNING is raised. Hard failures only happen for missing columns.
ALLOWED_NULL_MULTIPLIER = 3.0

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def schema_hash(features: list[str]) -> str:
    return hashlib.sha256(json.dumps(features).encode()).hexdigest()


def load_locked() -> dict:
    if not LOCKED_PATH.exists():
        log.error(f"features_locked.json not found: {LOCKED_PATH}")
        sys.exit(1)
    with open(LOCKED_PATH) as f:
        locked = json.load(f)
    # Verify hash integrity
    h = schema_hash(locked["features"])
    if h != locked["schema_hash"]:
        log.error("features_locked.json schema_hash mismatch — file is corrupt")
        sys.exit(1)
    return locked


def null_rate(df: pd.DataFrame, col: str) -> float:
    return float(df[col].isna().mean())


# ─────────────────────────────────────────────────────────────────────────────
# Per-split assertion
# ─────────────────────────────────────────────────────────────────────────────

def assert_split(
    split_name: str,
    df: pd.DataFrame,
    locked: dict,
    train_df: pd.DataFrame | None,
    train_null: dict[str, float] | None,
    strict: bool,
) -> tuple[int, int]:
    """
    Run all assertions on a single split.
    Returns (failures, warnings).
    """
    features   = locked["features"]
    fail_count = 0
    warn_count = 0

    log.info(f"  ── {split_name.upper()} ({'rows':>6}={len(df):,}  tickers={df['ticker'].nunique()}) ──")

    # ── 1. Column presence ────────────────────────────────────────────────
    missing = [f for f in features if f not in df.columns]
    if missing:
        fail_count += 1
        log.error(f"    [FAIL] Column presence: {len(missing)} locked features missing")
        for c in missing[:10]:
            log.error(f"      missing: {c}")
        if len(missing) > 10:
            log.error(f"      ... and {len(missing) - 10} more")
    else:
        log.info(f"    [OK  ] Column presence: all {len(features)} locked features present")

    # ── 2. Column order ───────────────────────────────────────────────────
    split_features_in_order = [c for c in df.columns if c in set(features)]
    if split_features_in_order != [f for f in features if f in set(split_features_in_order)]:
        warn_count += 1
        log.warning(f"    [WARN] Column order differs from locked schema")
    else:
        log.info(f"    [OK  ] Column order: consistent with locked schema")

    # ── 3. Dtype consistency (vs train) ──────────────────────────────────
    if train_df is not None:
        dtype_mismatches = []
        for f in features:
            if f not in df.columns:
                continue
            t_dtype = str(train_df[f].dtype)
            s_dtype = str(df[f].dtype)
            if t_dtype != s_dtype:
                # Allow float32/float64 interchange
                if set([t_dtype, s_dtype]) <= {"float32", "float64"}:
                    pass
                else:
                    dtype_mismatches.append((f, t_dtype, s_dtype))
        if dtype_mismatches:
            warn_count += 1
            log.warning(f"    [WARN] Dtype mismatches: {len(dtype_mismatches)}")
            for f, td, sd in dtype_mismatches[:5]:
                log.warning(f"      {f}: train={td}  {split_name}={sd}")
        else:
            log.info(f"    [OK  ] Dtypes: consistent with train")

    # ── 4. Target presence ────────────────────────────────────────────────
    missing_targets = [t for t in ["y_ret_21d", "y_dir_21d"] if t not in df.columns]
    if missing_targets:
        fail_count += 1
        log.error(f"    [FAIL] Target columns missing: {missing_targets}")
    else:
        y_null = df["y_ret_21d"].isna().mean()
        log.info(f"    [OK  ] Targets present  (y_ret_21d null={y_null:.1%})")

    # ── 5. Null budget check ──────────────────────────────────────────────
    if train_null is not None:
        null_violations = []
        for f in features:
            if f not in df.columns:
                continue
            s_null = null_rate(df, f)
            t_null = train_null.get(f, 0.0)
            budget = t_null * ALLOWED_NULL_MULTIPLIER
            # Only flag if the absolute increase is also meaningful (>5pp)
            if s_null > budget and (s_null - t_null) > 0.05:
                null_violations.append((f, t_null, s_null, budget))
        if null_violations:
            warn_count += 1
            log.warning(f"    [WARN] Null budget exceeded: {len(null_violations)} features")
            for f, tn, sn, bud in sorted(null_violations, key=lambda x: -x[2])[:8]:
                log.warning(f"      {f:<45} train={tn:.1%}  {split_name}={sn:.1%}  budget={bud:.1%}")
        else:
            log.info(f"    [OK  ] Null budget: all features within {ALLOWED_NULL_MULTIPLIER}× train null rate")

    # ── 6. Date monotonicity & no duplicate (date, ticker) ───────────────
    if "date" in df.columns and "ticker" in df.columns:
        dups = df.duplicated(subset=["date", "ticker"]).sum()
        if dups > 0:
            fail_count += 1
            log.error(f"    [FAIL] Duplicate (date, ticker) pairs: {dups:,}")
        else:
            log.info(f"    [OK  ] No duplicate (date, ticker) pairs")

        # Check monotonicity per ticker
        non_mono = 0
        for ticker, grp in df.groupby("ticker"):
            if not grp["date"].is_monotonic_increasing:
                non_mono += 1
        if non_mono > 0:
            fail_count += 1
            log.error(f"    [FAIL] Non-monotonic dates in {non_mono} tickers")
        else:
            log.info(f"    [OK  ] Date monotonicity: all tickers sorted")

    status = "PASS" if fail_count == 0 else "FAIL"
    log.info(f"    → {split_name.upper()}: {status}  (failures={fail_count}  warnings={warn_count})")
    return fail_count, warn_count


# ─────────────────────────────────────────────────────────────────────────────
# Split boundary check
# ─────────────────────────────────────────────────────────────────────────────

def assert_no_leakage(splits: dict[str, pd.DataFrame]) -> int:
    """Check that train end < val start < test start."""
    failures = 0
    dates = {}
    for name, df in splits.items():
        if "date" in df.columns:
            dates[name] = (df["date"].min(), df["date"].max())

    if "train" in dates and "val" in dates:
        if dates["train"][1] >= dates["val"][0]:
            log.error(f"  [FAIL] Leakage: train ends {dates['train'][1]} >= val starts {dates['val'][0]}")
            failures += 1
        else:
            log.info(f"  [OK  ] No train/val leakage  (train ends {dates['train'][1]}, val starts {dates['val'][0]})")

    if "val" in dates and "test" in dates:
        if dates["val"][1] >= dates["test"][0]:
            log.error(f"  [FAIL] Leakage: val ends {dates['val'][1]} >= test starts {dates['test'][0]}")
            failures += 1
        else:
            log.info(f"  [OK  ] No val/test leakage   (val ends {dates['val'][1]}, test starts {dates['test'][0]})")

    return failures


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def run(split_filter: str | None = None, strict: bool = False) -> None:
    log.info("=" * 65)
    log.info("assert_schema.py  Phase 3 Step 5")
    log.info(f"Locked config : {LOCKED_PATH}")
    log.info("=" * 65)

    locked = load_locked()
    features = locked["features"]
    log.info(f"Locked features : {len(features)}  |  Hash: {locked['schema_hash'][:16]}...")

    # Determine which splits to check
    splits_to_check = list(SPLIT_PATHS.keys())
    if split_filter:
        if split_filter not in SPLIT_PATHS:
            log.error(f"Unknown split: {split_filter}")
            sys.exit(1)
        splits_to_check = [split_filter]

    # Load all required splits
    loaded: dict[str, pd.DataFrame] = {}
    for name in splits_to_check:
        path = SPLIT_PATHS[name]
        if not path.exists():
            log.error(f"Split not found: {path}")
            sys.exit(1)
        log.info(f"Loading {name} ...")
        loaded[name] = pd.read_parquet(path)

    # Always need train for dtype/null reference — load if not already loaded
    if "train" not in loaded and "train" in SPLIT_PATHS:
        train_path = SPLIT_PATHS["train"]
        if train_path.exists():
            log.info("Loading train (for dtype/null reference) ...")
            loaded["train"] = pd.read_parquet(train_path)

    train_df   = loaded.get("train")
    train_null = {f: null_rate(train_df, f) for f in features if train_df is not None and f in train_df.columns}

    # Split boundary / leakage check
    log.info("Checking split boundaries (no leakage) ...")
    leakage_failures = assert_no_leakage(loaded)

    # Per-split assertions
    total_failures = leakage_failures
    total_warnings = 0

    for name in splits_to_check:
        df = loaded[name]
        f, w = assert_split(
            split_name=name,
            df=df,
            locked=locked,
            train_df=train_df if name != "train" else None,
            train_null=train_null if name != "train" else None,
            strict=strict,
        )
        total_failures += f
        total_warnings += w

    # Final verdict
    log.info("=" * 65)
    if total_failures == 0 and total_warnings == 0:
        log.info("SCHEMA ASSERTION COMPLETE — ALL CHECKS PASSED ✓")
        log.info("  Dataset is clean and ready for model training.")
    elif total_failures == 0:
        log.info(f"SCHEMA ASSERTION COMPLETE — PASSED WITH {total_warnings} WARNING(S)")
        if strict:
            log.error("  --strict mode: treating warnings as failures")
            sys.exit(1)
        else:
            log.info("  Dataset is usable. Review warnings before training.")
    else:
        log.error(f"SCHEMA ASSERTION FAILED — {total_failures} FAILURE(S)  {total_warnings} WARNING(S)")
        log.error("  Fix failures before proceeding to model training.")
        sys.exit(1)
    log.info("=" * 65)


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--split",  default=None,
                        help="Check only one split: train | val | test")
    parser.add_argument("--strict", action="store_true",
                        help="Treat warnings as failures (exit 1)")
    args = parser.parse_args()
    run(split_filter=args.split, strict=args.strict)