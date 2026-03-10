"""
train_dlinear.py
================
Phase 4 — DLinear training (multi-task: regression + classification).

DLinear (Zeng et al. 2023, "Are Transformers Effective for Time Series
Forecasting?") decomposes each feature's time series into:
  - Trend component  (via moving average smoothing)
  - Seasonal/residual component (original minus trend)
Then applies two independent linear layers (one per component) and sums.

WHY DLINEAR HERE
----------------
DLinear was designed for univariate time series. We adapt it for the
cross-sectional panel setting:
  - Input:  (batch, lookback, n_features)
  - For each feature independently: decompose + linear project
  - Concatenate all feature outputs → multi-task head

This "individual=True" variant applies separate weights per feature,
giving the model the ability to learn different trend/seasonal dynamics
for equity features vs macro features. When individual=False, weights
are shared across features (fewer params, faster, sometimes better).

The key value of DLinear is its inductive bias: it explicitly
models trend (slow macro regime changes) separately from the
seasonal/momentum component. This is well-suited to financial data
where both play distinct roles.

MULTI-TASK HEAD
---------------
After the DLinear backbone produces a feature vector (hidden_dim,),
two linear heads produce:
  - pred_ret:  scalar regression output (y_ret_21d)
  - pred_dir:  scalar logit for classification (y_dir_21d)

COLAB USAGE
-----------
    from google.colab import drive
    drive.mount('/content/drive')
    !python -m scripts.train.train_dlinear \\
        --drive-dir /content/drive/MyDrive/stockpred

USAGE
-----
    python -m scripts.train.train_dlinear
    python -m scripts.train.train_dlinear --drive-dir /content/drive/MyDrive/stockpred
    python -m scripts.train.train_dlinear --resume      # resume from checkpoint
    python -m scripts.train.train_dlinear --eval-only   # evaluate saved model
"""

import argparse
import logging
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.train.dataset import build_loaders, load_locked_features
from scripts.train.metrics import (
    MultiTaskLoss, evaluate_predictions, save_checkpoint, load_checkpoint
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)

CONFIG_PATH = PROJECT_ROOT / "config" / "train_config.yaml"


# ─────────────────────────────────────────────────────────────────────────────
# DLinear Architecture
# ─────────────────────────────────────────────────────────────────────────────

class MovingAvg(nn.Module):
    """Moving average smoothing for trend extraction."""
    def __init__(self, kernel_size: int, stride: int = 1):
        super().__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, L, C) → process per-channel
        # Pad ends to preserve sequence length
        front = x[:, 0:1, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        end   = x[:, -1:, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        x = torch.cat([front, x, end], dim=1)
        # Pool: (B, C, L) → (B, C, L) → (B, L, C)
        x = self.avg(x.permute(0, 2, 1)).permute(0, 2, 1)
        return x


class SeriesDecomposition(nn.Module):
    """Decompose into trend + seasonal/residual."""
    def __init__(self, kernel_size: int):
        super().__init__()
        self.moving_avg = MovingAvg(kernel_size)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        trend    = self.moving_avg(x)
        seasonal = x - trend
        return trend, seasonal


class DLinearBackbone(nn.Module):
    """
    DLinear backbone: decompose + linear project per feature.

    Args:
        lookback:    number of input time steps
        n_features:  number of input features
        hidden_dim:  output dimension (fed to task heads)
        individual:  if True, separate Linear per feature
        moving_avg:  kernel size for trend decomposition
        dropout:     dropout on the projected output
    """
    def __init__(
        self,
        lookback:    int,
        n_features:  int,
        hidden_dim:  int   = 256,
        individual:  bool  = False,
        moving_avg:  int   = 25,
        dropout:     float = 0.1,
    ):
        super().__init__()
        self.lookback   = lookback
        self.n_features = n_features
        self.hidden_dim = hidden_dim
        self.individual = individual

        self.decomp = SeriesDecomposition(moving_avg)

        if individual:
            # Separate linear per feature for trend and seasonal
            self.trend_proj    = nn.ModuleList(
                [nn.Linear(lookback, hidden_dim // n_features + 1) for _ in range(n_features)]
            )
            self.seasonal_proj = nn.ModuleList(
                [nn.Linear(lookback, hidden_dim // n_features + 1) for _ in range(n_features)]
            )
            self.out_dim = (hidden_dim // n_features + 1) * 2 * n_features
        else:
            # Shared linear across all features
            self.trend_proj    = nn.Linear(lookback * n_features, hidden_dim)
            self.seasonal_proj = nn.Linear(lookback * n_features, hidden_dim)
            self.out_dim = hidden_dim * 2

        self.proj_out = nn.Sequential(
            nn.Linear(self.out_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor, pad_mask: torch.Tensor | None = None) -> torch.Tensor:
        """
        Args:
            x:        (B, L, F) — input sequence
            pad_mask: (B, L) — True = real data, False = padding (unused by DLinear)
        Returns:
            (B, hidden_dim)
        """
        trend, seasonal = self.decomp(x)  # each (B, L, F)

        if self.individual:
            trend_out    = []
            seasonal_out = []
            for i in range(self.n_features):
                t = trend[:, :, i]       # (B, L)
                s = seasonal[:, :, i]    # (B, L)
                trend_out.append(self.trend_proj[i](t))
                seasonal_out.append(self.seasonal_proj[i](s))
            trend_out    = torch.cat(trend_out, dim=-1)    # (B, out)
            seasonal_out = torch.cat(seasonal_out, dim=-1) # (B, out)
        else:
            B = x.shape[0]
            trend_flat    = trend.reshape(B, -1)    # (B, L*F)
            seasonal_flat = seasonal.reshape(B, -1) # (B, L*F)
            trend_out    = self.trend_proj(trend_flat)
            seasonal_out = self.seasonal_proj(seasonal_flat)

        combined = torch.cat([trend_out, seasonal_out], dim=-1)  # (B, out_dim)
        return self.proj_out(combined)                           # (B, hidden_dim)


class DLinear(nn.Module):
    """
    Full DLinear model with multi-task heads.
    """
    def __init__(
        self,
        lookback:    int,
        n_features:  int,
        hidden_dim:  int   = 256,
        individual:  bool  = False,
        moving_avg:  int   = 25,
        dropout:     float = 0.1,
    ):
        super().__init__()
        self.backbone = DLinearBackbone(
            lookback, n_features, hidden_dim, individual, moving_avg, dropout
        )
        # Multi-task heads
        self.head_reg = nn.Linear(hidden_dim, 1)
        self.head_cls = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor, pad_mask: torch.Tensor | None = None):
        """
        Args:
            x:        (B, L, F)
            pad_mask: (B, L) optional
        Returns:
            pred_ret: (B, 1) — raw regression output
            pred_dir: (B, 1) — raw logit for direction
        """
        h        = self.backbone(x, pad_mask)
        pred_ret = self.head_reg(h)
        pred_dir = self.head_cls(h)
        return pred_ret, pred_dir


# ─────────────────────────────────────────────────────────────────────────────
# Training loop
# ─────────────────────────────────────────────────────────────────────────────

def run_epoch(
    model:      DLinear,
    loader,
    loss_fn:    MultiTaskLoss,
    optimizer:  optim.Optimizer | None,
    device:     torch.device,
    is_train:   bool,
) -> tuple[float, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    model.train(is_train)
    total_loss = 0.0
    n_batches  = 0

    all_pred_ret, all_pred_dir = [], []
    all_y_ret, all_y_dir       = [], []
    all_dates, all_tickers     = [], []

    ctx = torch.enable_grad() if is_train else torch.no_grad()
    with ctx:
        for batch in loader:
            x        = batch["x"].to(device)          # (B, L, F)
            pad_mask = batch["pad_mask"].to(device)    # (B, L)
            y_ret    = batch["y_ret"].to(device)       # (B,)
            y_dir    = batch["y_dir"].to(device)       # (B,)

            pred_ret, pred_dir = model(x, pad_mask)
            loss, loss_info    = loss_fn(pred_ret, pred_dir, y_ret, y_dir)

            if is_train:
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

            total_loss += loss_info["loss"]
            n_batches  += 1

            all_pred_ret.extend(pred_ret.squeeze(-1).detach().cpu().numpy().tolist())
            all_pred_dir.extend(torch.sigmoid(pred_dir.squeeze(-1)).detach().cpu().numpy().tolist())
            all_y_ret.extend(y_ret.cpu().numpy().tolist())
            all_y_dir.extend(y_dir.cpu().numpy().tolist())
            all_dates.extend(batch["date"])
            all_tickers.extend(batch["ticker"])

    avg_loss = total_loss / max(n_batches, 1)
    return (
        avg_loss,
        np.array(all_pred_ret),
        np.array(all_pred_dir),
        np.array(all_y_ret),
        np.array(all_y_dir),
        np.array(all_dates),
        np.array(all_tickers),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def run(drive_dir: str | None = None, resume: bool = False, eval_only: bool = False) -> None:
    log.info("=" * 65)
    log.info("train_dlinear.py  Phase 4")
    log.info("=" * 65)

    with open(CONFIG_PATH) as f:
        cfg = yaml.safe_load(f)

    if drive_dir:
        cfg["training"]["drive_checkpoint_dir"] = drive_dir

    model_cfg  = cfg["model_configs"]["dlinear"]
    train_cfg  = cfg["training"]
    loss_cfg   = cfg["loss"]
    eval_cfg   = cfg["evaluation"]

    # ── Device ────────────────────────────────────────────────────────────
    if torch.cuda.is_available():
        device = torch.device("cuda")
        log.info(f"Device: {torch.cuda.get_device_name(0)}")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        log.info("Device: Apple MPS")
    else:
        device = torch.device("cpu")
        log.info("Device: CPU")

    # ── Data ──────────────────────────────────────────────────────────────
    log.info("Building data loaders ...")
    train_loader, val_loader, test_loader = build_loaders(
        cfg, mode="sequence",
        num_workers=0,          # 0 avoids shared-memory errors in Colab
        pin_memory=(device.type == "cuda"),
    )

    features   = load_locked_features(cfg["data"]["locked_config"])
    n_features = len(features)
    lookback   = cfg["sequence"]["lookback"]

    # ── Model ─────────────────────────────────────────────────────────────
    model = DLinear(
        lookback   = lookback,
        n_features = n_features,
        hidden_dim = model_cfg["hidden_dim"],
        individual = model_cfg["individual"],
        moving_avg = model_cfg["moving_avg"],
        dropout    = model_cfg["dropout"],
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log.info(f"DLinear: {n_params:,} parameters  lookback={lookback}  features={n_features}")

    loss_fn   = MultiTaskLoss(loss_cfg["reg_weight"], loss_cfg["cls_weight"])
    optimizer = optim.AdamW(
        model.parameters(),
        lr=model_cfg["learning_rate"],
        weight_decay=model_cfg["weight_decay"],
    )
    warmup_epochs = 3
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs
        return 1.0
    warmup   = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    cosine   = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, model_cfg["max_epochs"] - warmup_epochs), eta_min=1e-6
    )
    scheduler = optim.lr_scheduler.SequentialLR(
        optimizer, schedulers=[warmup, cosine], milestones=[warmup_epochs]
    )

    # ── Resume from checkpoint ────────────────────────────────────────────
    start_epoch = 0
    best_val_ic = -999.0
    if resume or eval_only:
        model, optimizer, start_epoch, best_val_ic = load_checkpoint(
            model, optimizer, "dlinear", cfg, load_best=eval_only
        )

    if eval_only:
        log.info("--eval-only: skipping training ...")
    else:
        # ── Training loop ─────────────────────────────────────────────────
        patience_counter = 0
        max_epochs       = model_cfg["max_epochs"]
        early_stopping   = train_cfg["early_stopping"]

        log.info(f"Training for up to {max_epochs} epochs  "
                 f"(early stopping patience={early_stopping}) ...")

        for epoch in range(start_epoch, max_epochs):
            t0 = time.time()

            train_loss, _, _, _, _, _, _ = run_epoch(
                model, train_loader, loss_fn, optimizer, device, is_train=True
            )
            scheduler.step()

            # Val
            val_loss, pred_ret_v, pred_dir_v, y_ret_v, y_dir_v, dates_v, tickers_v = run_epoch(
                model, val_loader, loss_fn, None, device, is_train=False
            )
            val_eval = evaluate_predictions(
                pred_ret_v, pred_dir_v, y_ret_v, y_dir_v, dates_v, tickers_v,
                top_k_pct=eval_cfg["top_k_pct"],
                txn_cost=eval_cfg["transaction_cost"],
            )

            elapsed  = time.time() - t0
            is_best  = val_eval.ic_mean > best_val_ic
            if is_best:
                best_val_ic = val_eval.ic_mean
                patience_counter = 0
            else:
                patience_counter += 1

            log.info(
                f"Epoch {epoch+1:>4}/{max_epochs}  "
                f"train_loss={train_loss:.4f}  "
                f"val_loss={val_loss:.4f}  "
                f"val_IC={val_eval.ic_mean:+.4f}  "
                f"val_HitRate={val_eval.hit_rate:.1%}  "
                f"best_IC={best_val_ic:+.4f}  "
                f"patience={patience_counter}/{early_stopping}  "
                f"[{elapsed:.1f}s]"
            )

            save_checkpoint(model, optimizer, epoch, val_eval.ic_mean, cfg, "dlinear", is_best)

            if patience_counter >= early_stopping:
                log.info(f"Early stopping at epoch {epoch+1}")
                break

    # ── Final evaluation (load best model) ───────────────────────────────
    log.info("Loading best checkpoint for final evaluation ...")
    model, _, _, _ = load_checkpoint(model, None, "dlinear", cfg, load_best=True)
    model.eval()

    log.info("EVALUATING — VAL")
    _, pred_ret_v, pred_dir_v, y_ret_v, y_dir_v, dates_v, tickers_v = run_epoch(
        model, val_loader, loss_fn, None, device, is_train=False
    )
    val_results = evaluate_predictions(
        pred_ret_v, pred_dir_v, y_ret_v, y_dir_v, dates_v, tickers_v,
        top_k_pct=eval_cfg["top_k_pct"],
        txn_cost=eval_cfg["transaction_cost"],
    )

    log.info("EVALUATING — TEST")
    _, pred_ret_t, pred_dir_t, y_ret_t, y_dir_t, dates_t, tickers_t = run_epoch(
        model, test_loader, loss_fn, None, device, is_train=False
    )
    test_results = evaluate_predictions(
        pred_ret_t, pred_dir_t, y_ret_t, y_dir_t, dates_t, tickers_t,
        top_k_pct=eval_cfg["top_k_pct"],
        txn_cost=eval_cfg["transaction_cost"],
    )

    log.info("=" * 65)
    log.info("DLINEAR TRAINING COMPLETE")
    log.info(f"  Val  {val_results}")
    log.info(f"  Test {test_results}")
    log.info("=" * 65)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--drive-dir", type=str, default=None)
    parser.add_argument("--resume",    action="store_true")
    parser.add_argument("--eval-only", action="store_true")
    args = parser.parse_args()
    run(drive_dir=args.drive_dir, resume=args.resume, eval_only=args.eval_only)