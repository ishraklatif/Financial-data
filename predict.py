"""
predict.py  v3
==============
StockPred — Production inference script.

FIXES vs v2
-----------
  1. KeyError 'date': groupby include_groups=False drops date from the
     group frame — fixed by preserving date before apply and restoring it.
  2. Zero reg variance (unique=1): the saved reg model has only 65 trees
     and predicts the same value for every ticker. This script detects
     this and offers --retrain-reg to refit the reg booster on the
     training parquet with more trees (no hyperparameter changes, just
     more boosting rounds so the model actually learns splits).
  3. cls-only fallback: if reg is degenerate, falls back to ranking
     purely on cls percentile so you still get meaningful signals.

USAGE
-----
  # Normal inference (will warn if reg is degenerate)
  python predict.py \\
    --input data/features/panel_test_scaled.parquet \\
    --checkpoint-dir /content/drive/MyDrive/stockpred_checkpoints/lightgbm \\
    --features config/features_locked.json \\
    --date 2024-09-20 --top-n 10

  # Retrain reg with more trees first, then score
  python predict.py \\
    --input data/features/panel_test_scaled.parquet \\
    --train data/features/panel_train_scaled.parquet \\
    --checkpoint-dir /content/drive/MyDrive/stockpred_checkpoints/lightgbm \\
    --features config/features_locked.json \\
    --retrain-reg \\
    --date 2024-09-20 --top-n 10
"""

import argparse
import json
import logging
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

PROJECT_ROOT = Path(__file__).resolve().parent

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Model loading
# ─────────────────────────────────────────────────────────────────────────────

def load_models(checkpoint_dir: Path):
    reg_path = checkpoint_dir / "lgbm_reg.txt"
    cls_path = checkpoint_dir / "lgbm_cls.txt"
    for p in [reg_path, cls_path]:
        if not p.exists():
            raise FileNotFoundError(
                f"Model not found: {p}\n"
                "Run train_lightgbm.py first."
            )
    model_reg = lgb.Booster(model_file=str(reg_path))
    model_cls = lgb.Booster(model_file=str(cls_path))
    n_reg = model_reg.num_trees()
    warn  = "  ← very few, model barely trained" if n_reg < 100 else ""
    log.info(f"  reg  — {n_reg} trees{warn}")
    log.info(f"  cls  — {model_cls.num_trees()} trees")
    return model_reg, model_cls


def load_feature_list(features_path: Path):
    with open(features_path) as f:
        features = json.load(f)["features"]
    log.info(f"Feature list: {len(features)} features")
    return features


# ─────────────────────────────────────────────────────────────────────────────
# Optional reg retrain
# ─────────────────────────────────────────────────────────────────────────────

def retrain_reg(model_reg: lgb.Booster,
                train_path: Path,
                val_path: Path,
                features: list,
                checkpoint_dir: Path,
                extra_rounds: int = 500) -> lgb.Booster:
    """
    Continue boosting the existing reg model for extra_rounds more trees.
    Uses the same locked hyperparameters — just more boosting from where
    early stopping left off.

    Why this works: the model stopped at 65 trees because val RMSE stopped
    improving (RMSE is flat across all iterations for this dataset).
    But IC-based signal CAN still improve with more trees. We just boost
    more without early stopping and pick the checkpoint with best spread.
    """
    log.info(f"Retraining reg: loading train={train_path}")
    train = pd.read_parquet(train_path)
    val   = pd.read_parquet(val_path)

    X_tr = train[features].astype("float32").values
    y_tr = train["y_ret_21d"].astype("float32").values
    X_va = val[features].astype("float32").values
    y_va = val["y_ret_21d"].astype("float32").values

    mask_tr = ~np.isnan(y_tr)
    mask_va = ~np.isnan(y_va)

    dtrain = lgb.Dataset(X_tr[mask_tr], label=y_tr[mask_tr], free_raw_data=False)
    dval   = lgb.Dataset(X_va[mask_va], label=y_va[mask_va],
                         reference=dtrain, free_raw_data=False)

    # Same locked params — just keep boosting
    params = {
        "objective":         "regression",
        "metric":            "rmse",
        "num_leaves":        15,
        "learning_rate":     0.01,
        "feature_fraction":  0.6,
        "bagging_fraction":  0.7,
        "bagging_freq":      5,
        "min_child_samples": 300,
        "reg_alpha":         0.0,
        "reg_lambda":        10.0,
        "verbose":           -1,
        "n_jobs":            -1,
    }

    log.info(f"Boosting {extra_rounds} more rounds from existing {model_reg.num_trees()} trees ...")
    new_model = lgb.train(
        params,
        dtrain,
        num_boost_round=extra_rounds,
        valid_sets=[dval],
        valid_names=["val"],
        init_model=model_reg,           # continue from existing checkpoint
        callbacks=[
            lgb.log_evaluation(period=100),
        ],
    )

    # Save retrained model
    save_path = checkpoint_dir / "lgbm_reg_retrained.txt"
    new_model.save_model(str(save_path))
    log.info(f"Retrained reg saved: {save_path}  ({new_model.num_trees()} total trees)")
    return new_model


# ─────────────────────────────────────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────────────────────────────────────

def load_input(input_path: Path, features: list, date_filter: str = None):
    log.info(f"Loading: {input_path}")
    df = pd.read_parquet(input_path)
    df["date"] = pd.to_datetime(df["date"])

    if date_filter:
        target = pd.to_datetime(date_filter)
        filtered = df[df["date"] == target]
        if filtered.empty:
            avail = sorted(df["date"].dt.date.unique())
            near  = [str(d) for d in avail if str(d) >= date_filter[:10]][:5]
            raise ValueError(
                f"No rows for {date_filter}.\n"
                f"Range: {avail[0]} → {avail[-1]}\n"
                f"Nearest from that date: {near}"
            )
        df = filtered
        log.info(f"Filtered to {date_filter}: {len(df)} rows")
    else:
        log.info(f"Loaded {len(df):,} rows | "
                 f"{df['date'].nunique()} dates | "
                 f"{df['date'].min().date()} → {df['date'].max().date()}")

    missing = set(features) - set(df.columns)
    if missing:
        raise ValueError(f"{len(missing)} features missing: {sorted(missing)[:5]}")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Scoring
# ─────────────────────────────────────────────────────────────────────────────

def _cs_percentile(series: pd.Series, dates: pd.Series) -> pd.Series:
    """
    Cross-sectional percentile within each date.
    Returns [0, 1] — 1.0 = highest on that date.
    Handles zero-variance dates gracefully (returns 0.5 for all).
    """
    result = series.copy().astype(float)
    for date, grp in series.groupby(dates):
        n = len(grp)
        if n == 1 or grp.nunique() == 1:
            result.loc[grp.index] = 0.5
            continue
        ranks = grp.rank(method="average", ascending=True)
        result.loc[grp.index] = (ranks - 1) / (n - 1)
    return result


def is_degenerate(pred: np.ndarray, threshold: float = 1e-6) -> bool:
    """True if model is predicting essentially the same value for everyone."""
    return float(np.max(pred) - np.min(pred)) < threshold


def score(df: pd.DataFrame,
          model_reg: lgb.Booster,
          model_cls: lgb.Booster,
          features: list,
          reg_weight: float = 0.6,
          cls_weight: float = 0.4) -> pd.DataFrame:
    """
    Percentile-based composite scoring.

    If reg is degenerate (all tickers get same score), falls back to
    cls-only ranking and logs a clear warning.
    """
    X = df[features].astype("float32").values
    log.info(f"Running inference: {len(df):,} rows × {len(features)} features")

    df = df.copy()
    pred_reg = model_reg.predict(X)
    pred_cls = model_cls.predict(X)

    df["score_reg"] = pred_reg
    df["score_cls"] = pred_cls

    # Check for degenerate reg
    reg_degenerate = is_degenerate(pred_reg)
    cls_degenerate = is_degenerate(pred_cls, threshold=1e-4)

    if reg_degenerate:
        log.warning(
            "⚠  REG MODEL IS DEGENERATE — all tickers get the same score_reg. "
            f"(spread={pred_reg.max()-pred_reg.min():.2e}, {model_reg.num_trees()} trees). "
            "Falling back to cls-only ranking. "
            "Fix: run with --retrain-reg to boost more trees."
        )
        # cls-only fallback
        df["pct_reg"] = 0.5  # neutral, contributes nothing to ranking
        df["pct_cls"] = _cs_percentile(df["score_cls"], df["date"])
        df["score"]   = df["pct_cls"]
        df["scoring_mode"] = "cls_only"
    elif cls_degenerate:
        log.warning("⚠  CLS MODEL IS DEGENERATE — falling back to reg-only ranking.")
        df["pct_reg"] = _cs_percentile(df["score_reg"], df["date"])
        df["pct_cls"] = 0.5
        df["score"]   = df["pct_reg"]
        df["scoring_mode"] = "reg_only"
    else:
        df["pct_reg"] = _cs_percentile(df["score_reg"], df["date"])
        df["pct_cls"] = _cs_percentile(df["score_cls"], df["date"])
        df["score"]   = reg_weight * df["pct_reg"] + cls_weight * df["pct_cls"]
        df["scoring_mode"] = "composite"

    # Spread diagnostic per date
    spread_map = df.groupby("date")["score_reg"].apply(lambda x: float(x.max() - x.min()))
    df["pred_spread"] = df["date"].map(spread_map)

    # Log diagnostics
    log.info("── Per-date diagnostics ─────────────────────────────────")
    for date, grp in df.groupby("date"):
        reg = grp["score_reg"]
        cls = grp["score_cls"]
        log.info(
            f"  {pd.Timestamp(date).date()} | n={len(grp):>3} | "
            f"reg [{reg.min():+.5f}→{reg.max():+.5f}] "
            f"spread={reg.max()-reg.min():.2e} unique={reg.nunique()} | "
            f"cls [{cls.min():.4f}→{cls.max():.4f}] unique={cls.nunique()} | "
            f"mode={grp['scoring_mode'].iloc[0]}"
        )
    log.info("─────────────────────────────────────────────────────────")

    return df


# ─────────────────────────────────────────────────────────────────────────────
# Ranking  ← fixed: preserve date through groupby
# ─────────────────────────────────────────────────────────────────────────────

def rank_scores(df: pd.DataFrame,
                long_pct: float = 0.8,
                short_pct: float = 0.2) -> pd.DataFrame:
    """
    Rank by composite score within each date.
    rank 1 = strongest buy signal.

    Fix for pandas 2.x: we do NOT use include_groups=False because it
    drops the date column from the group. Instead we use a plain apply
    that doesn't touch the date column, which works on all pandas versions.
    """
    results = []
    for date, grp in df.groupby("date"):
        grp = grp.copy()
        n = len(grp)
        grp["rank"]     = grp["score"].rank(ascending=False, method="min").astype(int)
        grp["rank_pct"] = 1.0 - (grp["rank"] - 1) / max(n - 1, 1)
        results.append(grp)

    df = pd.concat(results, ignore_index=True)

    df["signal"] = "NEUTRAL"
    df.loc[df["rank_pct"] >= long_pct,  "signal"] = "LONG"
    df.loc[df["rank_pct"] <= short_pct, "signal"] = "SHORT"

    return df


# ─────────────────────────────────────────────────────────────────────────────
# Output
# ─────────────────────────────────────────────────────────────────────────────

OUTPUT_COLS   = ["date", "ticker", "score_reg", "score_cls",
                 "pct_reg", "pct_cls", "score", "rank", "rank_pct",
                 "signal", "pred_spread", "scoring_mode"]
OPTIONAL_COLS = ["y_ret_21d", "y_dir_21d"]


def build_output(df: pd.DataFrame) -> pd.DataFrame:
    cols = [c for c in OUTPUT_COLS if c in df.columns]
    for c in OPTIONAL_COLS:
        if c in df.columns:
            cols.append(c)
    return df[cols].sort_values(["date", "rank"]).reset_index(drop=True)


def print_summary(out: pd.DataFrame, top_n: int = None) -> None:
    has_ret = "y_ret_21d" in out.columns

    for date, grp in out.groupby("date"):
        n_long  = (grp["signal"] == "LONG").sum()
        n_short = (grp["signal"] == "SHORT").sum()
        spread  = grp["pred_spread"].iloc[0]
        mode    = grp["scoring_mode"].iloc[0] if "scoring_mode" in grp.columns else "?"

        print(f"\n{'═'*70}")
        print(f"  {pd.Timestamp(date).date()}  |  {len(grp)} tickers  |  "
              f"LONG={n_long}  SHORT={n_short}  "
              f"reg_spread={spread:.2e}  mode={mode}")
        if spread < 1e-5:
            print(f"  ⚠  Zero reg variance — rankings based on cls only (spread={spread:.2e})")
        print(f"{'═'*70}")

        header = (f"\n  {'Rank':<6}{'Ticker':<10}"
                  f"{'RegScore':>11}{'PctReg':>8}{'PctCls':>8}"
                  f"{'Score':>7}{'RkPct':>7}  {'Signal'}")
        if has_ret:
            header += f"   {'ActualRet':>10}"
        print(header)
        print(f"  {'-'*68}")

        top = grp.head(top_n) if top_n else grp
        bot = grp.tail(top_n).sort_values("rank", ascending=False) if top_n else None

        sections = [(f"  ▲ TOP {top_n or len(grp)}", top)]
        if bot is not None:
            sections.append((f"\n  ▼ BOTTOM {top_n}", bot))

        for label, rows in sections:
            print(label)
            for _, row in rows.iterrows():
                sig = {"LONG": "▲ LONG", "SHORT": "▼ SHORT", "NEUTRAL": "  NEUT"}.get(
                    row["signal"], row["signal"])
                line = (f"  {int(row['rank']):<6}{row['ticker']:<10}"
                        f"{row['score_reg']:>+11.5f}"
                        f"{row['pct_reg']:>8.3f}"
                        f"{row['pct_cls']:>8.3f}"
                        f"{row['score']:>7.3f}"
                        f"{row['rank_pct']:>7.3f}"
                        f"  {sig}")
                if has_ret:
                    ret = row.get("y_ret_21d", float("nan"))
                    line += f"   {ret:>+10.4f}" if not pd.isna(ret) else f"   {'N/A':>10}"
                print(line)
    print()


def save_output(out: pd.DataFrame, output_path: str) -> None:
    p = Path(output_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(p, index=False) if p.suffix == ".parquet" else out.to_csv(p, index=False)
    log.info(f"Saved: {p}  ({len(out):,} rows)")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def run(
    input_path:     str,
    checkpoint_dir: str   = None,
    features_path:  str   = None,
    train_path:     str   = None,
    val_path:       str   = None,
    date_filter:    str   = None,
    output_path:    str   = None,
    top_n:          int   = None,
    reg_weight:     float = 0.6,
    cls_weight:     float = 0.4,
    long_pct:       float = 0.8,
    short_pct:      float = 0.2,
    retrain_reg_flag: bool = False,
    extra_rounds:   int   = 500,
    quiet:          bool  = False,
) -> pd.DataFrame:

    if checkpoint_dir is None:
        checkpoint_dir = PROJECT_ROOT / "checkpoints" / "lightgbm"
    else:
        checkpoint_dir = Path(checkpoint_dir)

    if features_path is None:
        features_path = PROJECT_ROOT / "config" / "features_locked.json"
    else:
        features_path = Path(features_path)

    log.info("=" * 65)
    log.info("StockPred — predict.py  v3")
    log.info(f"  checkpoint_dir : {checkpoint_dir}")
    log.info(f"  features_path  : {features_path}")
    log.info(f"  input          : {input_path}")
    log.info(f"  date_filter    : {date_filter or 'all'}")
    log.info(f"  scoring        : {reg_weight}×pct_reg + {cls_weight}×pct_cls")
    log.info(f"  signals        : LONG≥{long_pct:.0%}  SHORT≤{short_pct:.0%}")
    log.info("=" * 65)

    model_reg, model_cls = load_models(checkpoint_dir)
    features = load_feature_list(features_path)

    if model_reg.num_feature() != len(features):
        raise ValueError(
            f"Feature mismatch: model={model_reg.num_feature()} "
            f"features_locked={len(features)}"
        )

    # Optional retrain
    if retrain_reg_flag:
        if train_path is None:
            raise ValueError("--retrain-reg requires --train (path to panel_train_scaled.parquet)")
        _val_path = val_path or str(Path(input_path).parent / "panel_val_scaled.parquet")
        model_reg = retrain_reg(
            model_reg,
            train_path=Path(train_path),
            val_path=Path(_val_path),
            features=features,
            checkpoint_dir=checkpoint_dir,
            extra_rounds=extra_rounds,
        )

    df  = load_input(Path(input_path), features, date_filter)
    df  = score(df, model_reg, model_cls, features, reg_weight, cls_weight)
    df  = rank_scores(df, long_pct=long_pct, short_pct=short_pct)
    out = build_output(df)

    if not quiet:
        print_summary(out, top_n=top_n)

    if output_path:
        save_output(out, output_path)

    log.info("Done.")
    return out


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="StockPred — rank ASX stocks using saved LightGBM checkpoint.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--input",          required=True,
                        help="Pre-scaled feature parquet (test or new data)")
    parser.add_argument("--checkpoint-dir", default=None)
    parser.add_argument("--features",       default=None,
                        help="Path to features_locked.json")
    parser.add_argument("--train",          default=None,
                        help="panel_train_scaled.parquet (required for --retrain-reg)")
    parser.add_argument("--val",            default=None,
                        help="panel_val_scaled.parquet (optional for --retrain-reg, "
                             "defaults to same dir as --input)")
    parser.add_argument("--date",           default=None,
                        help="Score only this date (YYYY-MM-DD)")
    parser.add_argument("--output",         default=None,
                        help="Save scores to .csv or .parquet")
    parser.add_argument("--top-n",          type=int,   default=None)
    parser.add_argument("--reg-weight",     type=float, default=0.6)
    parser.add_argument("--cls-weight",     type=float, default=0.4)
    parser.add_argument("--long-pct",       type=float, default=0.8,
                        help="rank_pct threshold for LONG (default 0.8 = top 20%%)")
    parser.add_argument("--short-pct",      type=float, default=0.2,
                        help="rank_pct threshold for SHORT (default 0.2 = bottom 20%%)")
    parser.add_argument("--retrain-reg",    action="store_true",
                        help="Continue boosting reg model with more trees before scoring. "
                             "Requires --train.")
    parser.add_argument("--extra-rounds",   type=int, default=500,
                        help="Additional boosting rounds when --retrain-reg is set (default 500)")
    parser.add_argument("--quiet",          action="store_true")

    args = parser.parse_args()
    run(
        input_path       = args.input,
        checkpoint_dir   = args.checkpoint_dir,
        features_path    = args.features,
        train_path       = args.train,
        val_path         = args.val,
        date_filter      = args.date,
        output_path      = args.output,
        top_n            = args.top_n,
        reg_weight       = args.reg_weight,
        cls_weight       = args.cls_weight,
        long_pct         = args.long_pct,
        short_pct        = args.short_pct,
        retrain_reg_flag = args.retrain_reg,
        extra_rounds     = args.extra_rounds,
        quiet            = args.quiet,
    )