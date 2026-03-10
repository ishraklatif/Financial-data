"""
metrics.py
==========
Phase 4 — Shared loss functions and evaluation metrics.

MULTI-TASK LOSS
---------------
    L = reg_weight * MaskedMSE(pred_ret, y_ret)
      + cls_weight * MaskedBCE(pred_dir, y_dir)

NaN targets are masked out of the loss computation.

EVALUATION METRICS
------------------
  IC       — Spearman rank correlation (cross-sectional, per date)
  ICIR     — IC / std(IC)
  Hit Rate — directional accuracy
  L/S Sharpe — annualised Sharpe of top-K long, bottom-K short portfolio
"""

import logging
import warnings
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import stats as scipy_stats

log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Multi-task loss
# ─────────────────────────────────────────────────────────────────────────────

class MaskedMSELoss(nn.Module):
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        mask = ~torch.isnan(target)
        if mask.sum() == 0:
            return torch.tensor(0.0, requires_grad=True, device=pred.device)
        return F.mse_loss(pred[mask], target[mask])


class MaskedBCELoss(nn.Module):
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        mask = ~torch.isnan(target)
        if mask.sum() == 0:
            return torch.tensor(0.0, requires_grad=True, device=pred.device)
        return F.binary_cross_entropy_with_logits(pred[mask], target[mask])


class MultiTaskLoss(nn.Module):
    def __init__(self, reg_weight: float = 0.6, cls_weight: float = 0.4):
        super().__init__()
        self.reg_weight = reg_weight
        self.cls_weight = cls_weight
        self.mse = MaskedMSELoss()
        self.bce = MaskedBCELoss()

    def forward(self, pred_ret, pred_dir, y_ret, y_dir):
        loss_reg = self.mse(pred_ret.squeeze(-1), y_ret)
        loss_cls = self.bce(pred_dir.squeeze(-1), y_dir)
        loss     = self.reg_weight * loss_reg + self.cls_weight * loss_cls
        return loss, {
            "loss":     loss.item(),
            "loss_reg": loss_reg.item(),
            "loss_cls": loss_cls.item(),
        }


# ─────────────────────────────────────────────────────────────────────────────
# Evaluation metrics
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class EvalResults:
    ic_mean:     float = 0.0
    ic_std:      float = 0.0
    icir:        float = 0.0
    ic_positive: float = 0.0
    hit_rate:    float = 0.0
    auc:         float = 0.5
    ls_sharpe:   float = 0.0
    ls_mean_ret: float = 0.0
    ls_std_ret:  float = 0.0
    top_k_pct:   float = 0.20
    ic_series:   list  = field(default_factory=list)
    dates:       list  = field(default_factory=list)

    def __str__(self):
        return (
            f"IC={self.ic_mean:+.4f}±{self.ic_std:.4f}  "
            f"ICIR={self.icir:+.3f}  "
            f"IC+%={self.ic_positive:.1%}  "
            f"HitRate={self.hit_rate:.1%}  "
            f"L/S Sharpe={self.ls_sharpe:+.3f}"
        )


def compute_ic_series(pred_ret, y_ret, dates):
    """Cross-sectional Spearman IC for each date. Returns (ic_values, unique_dates)."""
    unique_dates = np.unique(dates)
    ic_values    = []

    for date in unique_dates:
        mask  = dates == date
        p, y  = pred_ret[mask], y_ret[mask]
        valid = ~np.isnan(y) & ~np.isnan(p)

        if valid.sum() < 5:
            ic_values.append(np.nan)
            continue

        # Suppress ConstantInputWarning — handle it explicitly
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", scipy_stats.ConstantInputWarning
                                  if hasattr(scipy_stats, "ConstantInputWarning") else UserWarning)
            try:
                corr, _ = scipy_stats.spearmanr(p[valid], y[valid])
                ic_values.append(float(corr) if np.isfinite(corr) else np.nan)
            except Exception:
                ic_values.append(np.nan)

    return np.array(ic_values), unique_dates


def compute_longshort_sharpe(pred_ret, y_ret, dates, top_k_pct=0.20, txn_cost=0.001):
    """Long top_k_pct, short bottom_k_pct. Returns (annualised_sharpe, daily_returns)."""
    unique_dates  = np.unique(dates)
    daily_returns = []

    for date in unique_dates:
        mask  = dates == date
        p, y  = pred_ret[mask], y_ret[mask]
        valid = ~np.isnan(y) & ~np.isnan(p)

        if valid.sum() < 10:
            continue

        pv, yv = p[valid], y[valid]
        n      = len(pv)
        k      = max(1, int(n * top_k_pct))
        ranks  = np.argsort(pv)

        long_ret  = yv[ranks[-k:]].mean()
        short_ret = yv[ranks[:k]].mean()
        daily_returns.append((long_ret - short_ret) / 2.0 - txn_cost)

    if len(daily_returns) < 20:
        return 0.0, np.array(daily_returns)

    dr = np.array(daily_returns)
    sharpe = dr.mean() / (dr.std() + 1e-8) * np.sqrt(252 / 21)
    return float(sharpe), dr


def compute_hit_rate(pred_dir, y_dir) -> float:
    mask = ~np.isnan(y_dir) & ~np.isnan(pred_dir)
    if mask.sum() == 0:
        return 0.0
    pred_bin = (pred_dir[mask] > 0.5).astype(float)
    return float((pred_bin == y_dir[mask]).mean())


def evaluate_predictions(
    pred_ret, pred_dir, y_ret, y_dir, dates, tickers,
    top_k_pct=0.20, txn_cost=0.001,
) -> EvalResults:
    results = EvalResults(top_k_pct=top_k_pct)

    # IC series
    ic_vals, ic_dates = compute_ic_series(pred_ret, y_ret, dates)
    valid_ic = ic_vals[~np.isnan(ic_vals)]

    if len(valid_ic) > 0:
        results.ic_mean     = float(np.mean(valid_ic))
        results.ic_std      = float(np.std(valid_ic))
        results.icir        = float(results.ic_mean / (results.ic_std + 1e-8))
        results.ic_positive = float((valid_ic > 0).mean())
        results.ic_series   = valid_ic.tolist()
        results.dates       = [str(d) for d in ic_dates[~np.isnan(ic_vals)]]

    # Long-short Sharpe
    ls_sharpe, ls_rets = compute_longshort_sharpe(
        pred_ret, y_ret, dates, top_k_pct=top_k_pct, txn_cost=txn_cost
    )
    results.ls_sharpe   = ls_sharpe
    results.ls_mean_ret = float(ls_rets.mean()) if len(ls_rets) > 0 else 0.0
    results.ls_std_ret  = float(ls_rets.std())  if len(ls_rets) > 0 else 0.0

    # Hit rate
    results.hit_rate = compute_hit_rate(pred_dir, y_dir)

    # AUC
    try:
        from sklearn.metrics import roc_auc_score
        mask = ~np.isnan(y_dir) & ~np.isnan(pred_dir)
        if mask.sum() > 10 and len(np.unique(y_dir[mask])) > 1:
            results.auc = float(roc_auc_score(y_dir[mask], pred_dir[mask]))
    except Exception:
        results.auc = 0.5

    return results


# ─────────────────────────────────────────────────────────────────────────────
# Checkpoint utilities
# ─────────────────────────────────────────────────────────────────────────────

def save_checkpoint(model, optimizer, epoch, val_ic, cfg, model_name, is_best=False):
    from pathlib import Path
    import torch

    ckpt_dir = Path(cfg["training"]["checkpoint_dir"]) / model_name
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    state = {
        "epoch":     epoch,
        "val_ic":    val_ic,
        "model":     model.state_dict(),
        "optimizer": optimizer.state_dict(),
    }

    torch.save(state, ckpt_dir / "latest.pt")

    if is_best:
        torch.save(state, ckpt_dir / "best.pt")
        log.info(f"  [BEST] epoch={epoch}  val_IC={val_ic:.4f}")

    # Sync to Drive
    drive_dir = cfg["training"].get("drive_checkpoint_dir")
    if drive_dir:
        import shutil
        dp = Path(drive_dir) / model_name
        dp.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ckpt_dir / "latest.pt", dp / "latest.pt")
        if is_best:
            shutil.copy2(ckpt_dir / "best.pt", dp / "best.pt")


def load_checkpoint(model, optimizer, model_name, cfg, load_best=True):
    from pathlib import Path
    import torch

    fname = "best.pt" if load_best else "latest.pt"
    ckpt_path = None

    drive_dir = cfg["training"].get("drive_checkpoint_dir")
    if drive_dir:
        dp = Path(drive_dir) / model_name / fname
        if dp.exists():
            ckpt_path = dp

    if ckpt_path is None:
        lp = Path(cfg["training"]["checkpoint_dir"]) / model_name / fname
        if lp.exists():
            ckpt_path = lp

    if ckpt_path is None:
        log.info(f"No checkpoint found for {model_name} — starting fresh")
        return model, optimizer, 0, -999.0

    log.info(f"Loading checkpoint: {ckpt_path}")
    state = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(state["model"])
    if optimizer is not None:
        optimizer.load_state_dict(state["optimizer"])

    log.info(f"  Resumed epoch={state['epoch']}  val_IC={state['val_ic']:.4f}")
    return model, optimizer, state["epoch"] + 1, state["val_ic"]
