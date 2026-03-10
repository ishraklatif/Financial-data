"""
train_ft_transformer.py
=======================
Phase 4 — FT-Transformer training (Feature Tokenization Transformer).

Gorishniy et al. 2021, "Revisiting Deep Learning Models for Tabular Data."
FT-Transformer tokenises each tabular feature into a d_token embedding,
then applies standard Transformer blocks (multi-head attention + FFN).

WHY FT-TRANSFORMER
------------------
Unlike PatchTST (which patches time sequences), FT-Transformer operates
on the feature dimension — each of the 260 features becomes a token.
This gives the model attention over feature interactions:
  - Which macro features are currently most important?
  - How does the yield curve interact with equity momentum?
  - Does VIX affect energy stocks differently to banks?

This attention-over-features pattern is the key capability gap between
LightGBM (which learns feature splits, not interactions as soft weights)
and FT-Transformer.

TOKENISATION
------------
Each feature x_i gets a unique weight vector w_i ∈ R^d_token plus a
shared bias. The token for feature i is:
    t_i = x_i * w_i + b   (element-wise: scalar * vector + vector)
This allows the model to learn a distinct semantic embedding per feature.
A [CLS] token is prepended; its final representation is fed to the heads.

MULTI-TASK HEAD
---------------
    [CLS] representation → LayerNorm →
        head_reg → pred_ret (scalar)
        head_cls → pred_dir (logit)

COLAB USAGE
-----------
    from google.colab import drive
    drive.mount('/content/drive')
    !python -m scripts.train.train_ft_transformer \\
        --drive-dir /content/drive/MyDrive/stockpred

USAGE
-----
    python -m scripts.train.train_ft_transformer
    python -m scripts.train.train_ft_transformer --drive-dir /path/to/drive
    python -m scripts.train.train_ft_transformer --resume
    python -m scripts.train.train_ft_transformer --eval-only
"""

import argparse
import logging
import math
import sys
import time
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
# FT-Transformer Architecture
# ─────────────────────────────────────────────────────────────────────────────

class FeatureTokenizer(nn.Module):
    """
    Tokenise each scalar feature into a d_token-dimensional embedding.
    Handles NaN-masked features via the input mask.

    x_i → t_i = x_i * W_i + B_i   (W_i, B_i learned per feature)

    A [CLS] token is prepended to the sequence.
    """
    def __init__(self, n_features: int, d_token: int):
        super().__init__()
        self.n_features = n_features
        self.d_token    = d_token

        # Per-feature weight and bias vectors
        self.weight = nn.Parameter(torch.empty(n_features, d_token))
        self.bias   = nn.Parameter(torch.zeros(n_features, d_token))

        # [CLS] token embedding (learned, shape: d_token)
        self.cls_token = nn.Parameter(torch.empty(1, d_token))

        # Initialise
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        nn.init.normal_(self.cls_token, std=0.01)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        """
        Args:
            x:    (B, F) — scaled feature values, 0-imputed where NaN
            mask: (B, F) — True where feature is valid (non-NaN). Optional.
        Returns:
            tokens: (B, F+1, d_token)  — [CLS, feat_0, feat_1, ...]
        """
        B = x.shape[0]
        # x: (B, F) → (B, F, 1) * (F, d_token) → (B, F, d_token)
        tokens = x.unsqueeze(-1) * self.weight.unsqueeze(0) + self.bias.unsqueeze(0)

        # Zero out NaN-masked tokens
        if mask is not None:
            tokens = tokens * mask.unsqueeze(-1).float()

        # Prepend [CLS]
        cls = self.cls_token.unsqueeze(0).expand(B, -1, -1)  # (B, 1, d_token)
        tokens = torch.cat([cls, tokens], dim=1)              # (B, F+1, d_token)
        return tokens


class TransformerBlock(nn.Module):
    """
    Pre-LN Transformer block: LayerNorm → Attention → residual,
    LayerNorm → FFN → residual.
    """
    def __init__(
        self,
        d_token:           int,
        n_heads:           int,
        ffn_factor:        float = 1.333,
        attention_dropout: float = 0.2,
        ffn_dropout:       float = 0.1,
        residual_dropout:  float = 0.0,
    ):
        super().__init__()
        d_ffn = int(d_token * ffn_factor)

        self.norm1 = nn.LayerNorm(d_token)
        self.attn  = nn.MultiheadAttention(
            d_token, n_heads,
            dropout=attention_dropout,
            batch_first=True,
        )
        self.norm2 = nn.LayerNorm(d_token)
        self.ffn   = nn.Sequential(
            nn.Linear(d_token, d_ffn),
            nn.GELU(),
            nn.Dropout(ffn_dropout),
            nn.Linear(d_ffn, d_token),
        )
        self.drop_res = nn.Dropout(residual_dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Pre-LN attention
        h = self.norm1(x)
        h, _ = self.attn(h, h, h)
        x = x + self.drop_res(h)
        # Pre-LN FFN
        h = self.norm2(x)
        h = self.ffn(h)
        x = x + self.drop_res(h)
        return x


class FTTransformer(nn.Module):
    """
    Full FT-Transformer with multi-task output heads.

    Architecture:
        FeatureTokenizer → N × TransformerBlock → [CLS] → heads
    """
    def __init__(
        self,
        n_features:        int,
        d_token:           int   = 96,
        n_blocks:          int   = 2,
        n_heads:           int   = 6,
        ffn_factor:        float = 1.333,
        attention_dropout: float = 0.2,
        ffn_dropout:       float = 0.1,
        residual_dropout:  float = 0.0,
    ):
        super().__init__()
        self.tokenizer = FeatureTokenizer(n_features, d_token)
        self.blocks    = nn.ModuleList([
            TransformerBlock(
                d_token, n_heads, ffn_factor,
                attention_dropout, ffn_dropout, residual_dropout
            )
            for _ in range(n_blocks)
        ])
        self.norm_out  = nn.LayerNorm(d_token)

        # Multi-task heads from [CLS] representation
        self.head_reg  = nn.Sequential(
            nn.Linear(d_token, d_token // 2),
            nn.GELU(),
            nn.Linear(d_token // 2, 1),
        )
        self.head_cls  = nn.Sequential(
            nn.Linear(d_token, d_token // 2),
            nn.GELU(),
            nn.Linear(d_token // 2, 1),
        )

    def forward(
        self,
        x:    torch.Tensor,                # (B, F)
        mask: torch.Tensor | None = None,  # (B, F) True = valid
    ) -> tuple[torch.Tensor, torch.Tensor]:
        tokens   = self.tokenizer(x, mask)         # (B, F+1, d_token)
        for block in self.blocks:
            tokens = block(tokens)
        cls_repr = self.norm_out(tokens[:, 0, :])  # (B, d_token) — [CLS] position
        return self.head_reg(cls_repr), self.head_cls(cls_repr)


# ─────────────────────────────────────────────────────────────────────────────
# Training loop
# ─────────────────────────────────────────────────────────────────────────────

def run_epoch(
    model:     FTTransformer,
    loader,
    loss_fn:   MultiTaskLoss,
    optimizer: optim.Optimizer | None,
    device:    torch.device,
    is_train:  bool,
) -> tuple:
    model.train(is_train)
    total_loss = 0.0
    n_batches  = 0

    all_pred_ret, all_pred_dir = [], []
    all_y_ret, all_y_dir       = [], []
    all_dates, all_tickers     = [], []

    ctx = torch.enable_grad() if is_train else torch.no_grad()
    with ctx:
        for batch in loader:
            x     = batch["x"].to(device)       # (B, F)
            mask  = batch["mask"].to(device)     # (B, F)
            y_ret = batch["y_ret"].to(device)
            y_dir = batch["y_dir"].to(device)

            pred_ret, pred_dir = model(x, mask)
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

    return (
        total_loss / max(n_batches, 1),
        np.array(all_pred_ret), np.array(all_pred_dir),
        np.array(all_y_ret),    np.array(all_y_dir),
        np.array(all_dates),    np.array(all_tickers),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def run(drive_dir: str | None = None, resume: bool = False, eval_only: bool = False) -> None:
    log.info("=" * 65)
    log.info("train_ft_transformer.py  Phase 4")
    log.info("=" * 65)

    with open(CONFIG_PATH) as f:
        cfg = yaml.safe_load(f)

    if drive_dir:
        cfg["training"]["drive_checkpoint_dir"] = drive_dir

    model_cfg = cfg["model_configs"]["ft_transformer"]
    train_cfg = cfg["training"]
    loss_cfg  = cfg["loss"]
    eval_cfg  = cfg["evaluation"]

    # ── Device ────────────────────────────────────────────────────────────
    device = (torch.device("cuda") if torch.cuda.is_available()
              else torch.device("mps") if torch.backends.mps.is_available()
              else torch.device("cpu"))
    log.info(f"Device: {device}" +
             (f" ({torch.cuda.get_device_name(0)})" if device.type == "cuda" else ""))

    # ── Data ──────────────────────────────────────────────────────────────
    log.info("Building data loaders ...")
    train_loader, val_loader, test_loader = build_loaders(
        cfg, mode="tabular",
        num_workers=0,          # 0 avoids shared-memory errors in Colab
        pin_memory=(device.type == "cuda"),
    )
    features   = load_locked_features(cfg["data"]["locked_config"])
    n_features = len(features)

    # ── Model ─────────────────────────────────────────────────────────────
    model = FTTransformer(
        n_features        = n_features,
        d_token           = model_cfg["d_token"],
        n_blocks          = model_cfg["n_blocks"],
        n_heads           = model_cfg["attention_heads"],
        ffn_factor        = model_cfg["ffn_factor"],
        attention_dropout = model_cfg["attention_dropout"],
        ffn_dropout       = model_cfg["ffn_dropout"],
        residual_dropout  = model_cfg["residual_dropout"],
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log.info(f"FT-Transformer: {n_params:,} parameters  features={n_features}  "
             f"d_token={model_cfg['d_token']}  blocks={model_cfg['n_blocks']}  "
             f"heads={model_cfg['attention_heads']}")

    loss_fn   = MultiTaskLoss(loss_cfg["reg_weight"], loss_cfg["cls_weight"])
    optimizer = optim.AdamW(
        model.parameters(),
        lr=model_cfg["learning_rate"],
        weight_decay=model_cfg["weight_decay"],
    )

    # Linear warmup + cosine decay
    warmup_epochs = model_cfg.get("warmup_epochs", 3)
    max_epochs    = model_cfg["max_epochs"]

    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs
        progress = (epoch - warmup_epochs) / max(1, max_epochs - warmup_epochs)
        return 0.5 * (1 + math.cos(math.pi * progress))

    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # ── Resume ────────────────────────────────────────────────────────────
    start_epoch, best_val_ic = 0, -999.0
    if resume or eval_only:
        model, optimizer, start_epoch, best_val_ic = load_checkpoint(
            model, optimizer, "ft_transformer", cfg, load_best=eval_only
        )

    if not eval_only:
        patience_counter = 0
        early_stopping   = train_cfg["early_stopping"]

        log.info(f"Training for up to {max_epochs} epochs  "
                 f"(early stopping patience={early_stopping}) ...")

        for epoch in range(start_epoch, max_epochs):
            t0 = time.time()

            train_loss, *_ = run_epoch(
                model, train_loader, loss_fn, optimizer, device, is_train=True
            )
            scheduler.step()

            val_loss, pred_ret_v, pred_dir_v, y_ret_v, y_dir_v, dates_v, tickers_v = run_epoch(
                model, val_loader, loss_fn, None, device, is_train=False
            )
            val_eval = evaluate_predictions(
                pred_ret_v, pred_dir_v, y_ret_v, y_dir_v, dates_v, tickers_v,
                top_k_pct=eval_cfg["top_k_pct"], txn_cost=eval_cfg["transaction_cost"],
            )

            is_best = val_eval.ic_mean > best_val_ic
            if is_best:
                best_val_ic      = val_eval.ic_mean
                patience_counter = 0
            else:
                patience_counter += 1

            lr_now = optimizer.param_groups[0]["lr"]
            log.info(
                f"Epoch {epoch+1:>4}/{max_epochs}  "
                f"train_loss={train_loss:.4f}  val_IC={val_eval.ic_mean:+.4f}  "
                f"HitRate={val_eval.hit_rate:.1%}  "
                f"best={best_val_ic:+.4f}  "
                f"patience={patience_counter}/{early_stopping}  "
                f"lr={lr_now:.2e}  [{time.time()-t0:.1f}s]"
            )

            save_checkpoint(model, optimizer, epoch, val_eval.ic_mean,
                            cfg, "ft_transformer", is_best)

            if patience_counter >= early_stopping:
                log.info(f"Early stopping at epoch {epoch+1}")
                break

    # ── Final evaluation ──────────────────────────────────────────────────
    log.info("Loading best checkpoint for final evaluation ...")
    model, _, _, _ = load_checkpoint(model, None, "ft_transformer", cfg, load_best=True)
    model.eval()

    for split_name, loader in [("val", val_loader), ("test", test_loader)]:
        log.info(f"EVALUATING — {split_name.upper()}")
        _, pr, pd_, yr, yd, dt, tk = run_epoch(
            model, loader, loss_fn, None, device, is_train=False
        )
        results = evaluate_predictions(
            pr, pd_, yr, yd, dt, tk,
            top_k_pct=eval_cfg["top_k_pct"], txn_cost=eval_cfg["transaction_cost"],
        )
        log.info(f"  {split_name.upper()}  {results}")

    log.info("=" * 65)
    log.info("FT-TRANSFORMER TRAINING COMPLETE")
    log.info("=" * 65)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--drive-dir", type=str, default=None)
    parser.add_argument("--resume",    action="store_true")
    parser.add_argument("--eval-only", action="store_true")
    args = parser.parse_args()
    run(drive_dir=args.drive_dir, resume=args.resume, eval_only=args.eval_only)