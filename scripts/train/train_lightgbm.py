"""
train_lightgbm.py
=================
Phase 4 — LightGBM training (regression + classification).

Trains two independent models:
  lgbm_reg — predicts y_ret_21d (21-day forward log return)
  lgbm_cls — predicts y_dir_21d (probability of positive return)

Assumes panel_train_scaled.parquet is pre-scaled. No runtime scaling applied.

USAGE
-----
    python -m scripts.train.train_lightgbm
    python -m scripts.train.train_lightgbm --drive-dir /content/drive/MyDrive/stockpred_checkpoints
    python -m scripts.train.train_lightgbm --eval-only
"""

import argparse
import json
import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
import lightgbm as lgb

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.train.metrics import evaluate_predictions

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)
CONFIG_PATH = PROJECT_ROOT / "config" / "train_config.yaml"


# ─────────────────────────────────────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────────────────────────────────────

def load_split(cfg: dict, split: str):
    """Load a split. Returns (X, y_ret, y_dir, dates, tickers). No scaling applied."""
    data_cfg = cfg["data"]
    with open(data_cfg["locked_config"]) as f:
        features = json.load(f)["features"]

    path = {"train": data_cfg["train_path"],
             "val":   data_cfg["val_path"],
             "test":  data_cfg["test_path"]}[split]

    df = pd.read_parquet(path)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["ticker", "date"]).reset_index(drop=True)

    X       = df[features].astype("float32").values   # NaN preserved for LightGBM
    y_ret   = df["y_ret_21d"].astype("float32").values
    y_dir   = df["y_dir_21d"].astype("float32").values
    dates   = df["date"].astype(str).values
    tickers = df["ticker"].values

    valid = ~np.isnan(y_ret)
    log.info(f"  {split}: {len(df):,} rows  |  {valid.sum():,} valid targets  |  features={len(features)}")
    return X, y_ret, y_dir, dates, tickers


# ─────────────────────────────────────────────────────────────────────────────
# Training
# ─────────────────────────────────────────────────────────────────────────────

def train_regression(X_train, y_train, X_val, y_val, cfg) -> lgb.Booster:
    p          = cfg["model_configs"]["lightgbm"]["reg"]
    train_mask = ~np.isnan(y_train)
    val_mask   = ~np.isnan(y_val)

    dtrain = lgb.Dataset(X_train[train_mask], label=y_train[train_mask], free_raw_data=False)
    dval   = lgb.Dataset(X_val[val_mask],     label=y_val[val_mask],
                         reference=dtrain,    free_raw_data=False)

    params = {
        "objective":         "regression",
        "metric":            "rmse",
        "num_leaves":        p.get("num_leaves", 15),
        "max_depth":         p.get("max_depth", -1),
        "learning_rate":     p.get("learning_rate", 0.01),
        "feature_fraction":  p.get("feature_fraction", 0.6),
        "bagging_fraction":  p.get("bagging_fraction", 0.7),
        "bagging_freq":      p.get("bagging_freq", 5),
        "min_child_samples": p.get("min_child_samples", 300),
        "reg_alpha":         p.get("reg_alpha", 0.0),
        "reg_lambda":        p.get("reg_lambda", 10.0),
        "verbose":           -1,
        "n_jobs":            -1,
    }

    log.info(f"Training regression  train={train_mask.sum():,}  val={val_mask.sum():,}  features={X_train.shape[1]}")
    model = lgb.train(
        params, dtrain,
        num_boost_round=p.get("n_estimators", 3000),
        valid_sets=[dtrain, dval],
        valid_names=["train", "val"],
        callbacks=[
            lgb.early_stopping(p.get("early_stopping_rounds", 300), verbose=False),
            lgb.log_evaluation(period=100),
        ],
    )
    log.info(f"  Best iteration: {model.best_iteration}  val_rmse={model.best_score['val']['rmse']:.6f}")
    return model


def train_classification(X_train, y_train, X_val, y_val, cfg) -> lgb.Booster:
    p          = cfg["model_configs"]["lightgbm"]["cls"]
    train_mask = ~np.isnan(y_train)
    val_mask   = ~np.isnan(y_val)

    dtrain = lgb.Dataset(X_train[train_mask], label=y_train[train_mask], free_raw_data=False)
    dval   = lgb.Dataset(X_val[val_mask],     label=y_val[val_mask],
                         reference=dtrain,    free_raw_data=False)

    params = {
        "objective":         "binary",
        "metric":            "binary_logloss",
        "num_leaves":        p.get("num_leaves", 63),
        "max_depth":         p.get("max_depth", -1),
        "learning_rate":     p.get("learning_rate", 0.01),
        "feature_fraction":  p.get("feature_fraction", 0.6),
        "bagging_fraction":  p.get("bagging_fraction", 0.7),
        "bagging_freq":      p.get("bagging_freq", 5),
        "min_child_samples": p.get("min_child_samples", 100),
        "reg_alpha":         p.get("reg_alpha", 0.0),
        "reg_lambda":        p.get("reg_lambda", 1.0),
        "verbose":           -1,
        "n_jobs":            -1,
    }

    log.info(f"Training classification  train={train_mask.sum():,}  val={val_mask.sum():,}")
    model = lgb.train(
        params, dtrain,
        num_boost_round=p.get("n_estimators", 3000),
        valid_sets=[dtrain, dval],
        valid_names=["train", "val"],
        callbacks=[
            lgb.early_stopping(p.get("early_stopping_rounds", 300), verbose=False),
            lgb.log_evaluation(period=100),
        ],
    )
    log.info(f"  Best iteration: {model.best_iteration}  val_logloss={model.best_score['val']['binary_logloss']:.6f}")
    return model


# ─────────────────────────────────────────────────────────────────────────────
# Utilities
# ─────────────────────────────────────────────────────────────────────────────

def log_feature_importance(model: lgb.Booster, features: list, n: int = 20) -> None:
    importance = model.feature_importance(importance_type="gain")
    top = sorted(zip(features, importance), key=lambda x: -x[1])[:n]
    log.info(f"Top {n} features by gain:")
    for i, (name, gain) in enumerate(top):
        log.info(f"  {i+1:>3}. {name:<45}  gain={gain:.1f}")


def save_models(model_reg, model_cls, ckpt_dir: Path, drive_dir, results: dict) -> None:
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    model_reg.save_model(str(ckpt_dir / "lgbm_reg.txt"))
    model_cls.save_model(str(ckpt_dir / "lgbm_cls.txt"))
    with open(ckpt_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)
    log.info(f"Saved to {ckpt_dir}")

    if drive_dir:
        import shutil
        dp = Path(drive_dir) / "lightgbm"
        dp.mkdir(parents=True, exist_ok=True)
        for fname in ["lgbm_reg.txt", "lgbm_cls.txt", "results.json"]:
            shutil.copy2(ckpt_dir / fname, dp / fname)
        log.info(f"Synced to Drive: {dp}")


def load_models(ckpt_dir: Path, drive_dir) -> tuple:
    src = Path(drive_dir) / "lightgbm" if drive_dir and (Path(drive_dir) / "lightgbm" / "lgbm_reg.txt").exists() else ckpt_dir
    log.info(f"Loading from {src}")
    return lgb.Booster(model_file=str(src / "lgbm_reg.txt")), \
           lgb.Booster(model_file=str(src / "lgbm_cls.txt"))


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def run(drive_dir=None, eval_only=False):
    log.info("=" * 65)
    log.info("train_lightgbm.py  Phase 4")
    log.info(f"Drive dir: {drive_dir or 'local only'}")
    log.info("=" * 65)

    with open(CONFIG_PATH) as f:
        cfg = yaml.safe_load(f)

    log.info("Loading data ...")
    X_tr, y_ret_tr, y_dir_tr, dates_tr, tickers_tr = load_split(cfg, "train")
    X_va, y_ret_va, y_dir_va, dates_va, tickers_va = load_split(cfg, "val")
    X_te, y_ret_te, y_dir_te, dates_te, tickers_te = load_split(cfg, "test")

    ckpt_dir = PROJECT_ROOT / cfg["training"]["checkpoint_dir"] / "lightgbm"

    if eval_only:
        model_reg, model_cls = load_models(ckpt_dir, drive_dir)
    else:
        t0 = time.time()
        model_reg = train_regression(X_tr, y_ret_tr, X_va, y_ret_va, cfg)
        log.info(f"Regression time: {(time.time()-t0)/60:.1f} min")

        t0 = time.time()
        model_cls = train_classification(X_tr, y_dir_tr, X_va, y_dir_va, cfg)
        log.info(f"Classification time: {(time.time()-t0)/60:.1f} min")

    with open(cfg["data"]["locked_config"]) as f:
        features = json.load(f)["features"]
    log_feature_importance(model_reg, features)

    results = {"model": "lightgbm", "generated_at": datetime.now(timezone.utc).isoformat()}

    for split_name, X, y_ret, y_dir, dates, tickers in [
        ("val",  X_va, y_ret_va, y_dir_va, dates_va, tickers_va),
        ("test", X_te, y_ret_te, y_dir_te, dates_te, tickers_te),
    ]:
        log.info("=" * 65)
        log.info(f"EVALUATING — {split_name.upper()}")
        pred_ret = model_reg.predict(X)
        pred_dir = model_cls.predict(X)
        ev = evaluate_predictions(
            pred_ret, pred_dir, y_ret, y_dir, dates, tickers,
            top_k_pct=cfg["evaluation"]["top_k_pct"],
            txn_cost=cfg["evaluation"]["transaction_cost"],
        )
        log.info(f"  {split_name.upper()}  {ev}")
        results[split_name] = {
            "ic_mean": ev.ic_mean, "ic_std": ev.ic_std, "icir": ev.icir,
            "ic_positive": ev.ic_positive, "hit_rate": ev.hit_rate,
            "auc": ev.auc, "ls_sharpe": ev.ls_sharpe,
        }

    if not eval_only:
        save_models(model_reg, model_cls, ckpt_dir, drive_dir, results)

    log.info("=" * 65)
    log.info("LIGHTGBM COMPLETE")
    log.info(f"  Val  IC={results['val']['ic_mean']:+.4f}  L/S Sharpe={results['val']['ls_sharpe']:+.3f}")
    log.info(f"  Test IC={results['test']['ic_mean']:+.4f}  L/S Sharpe={results['test']['ls_sharpe']:+.3f}")
    log.info("=" * 65)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--drive-dir", type=str, default=None)
    parser.add_argument("--eval-only", action="store_true")
    args = parser.parse_args()
    run(drive_dir=args.drive_dir, eval_only=args.eval_only)