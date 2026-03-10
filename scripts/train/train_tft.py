"""
train_tft.py
============
Phase 4 — Temporal Fusion Transformer (TFT) training.

Bryan Lim et al. 2021, "Temporal Fusion Transformers for Interpretable
Multi-horizon Time Series Forecasting."

FOR THIS IMPLEMENTATION
-----------------------
Memory-efficient TFT that:
  - Replaces VSN with a single Linear(n_features → d_model) + LayerNorm.
    The original VSN (260 GRNs × 126 timesteps) OOMs on T4 regardless of
    hidden size. The linear projection preserves the encoder→LSTM→attention
    structure while fitting comfortably in 15 GB VRAM.
  - Uses a 21-day lookback (one trading month) instead of 126.
    Reduces sequence-dimension memory cost by 6× and keeps the model
    focused on short-term momentum signals.
  - Keeps GRN, LSTM, multi-head attention, static ticker embedding,
    and dual regression+classification heads (the core TFT components).

LOOKBACK CHANGE
---------------
train_config.yaml sequence.lookback must be set to 21 for this script.

COLAB USAGE — shell (recommended)
----------------------------------
    from google.colab import drive
    drive.mount('/content/drive')
    import os; os.chdir('/content/drive/MyDrive/stockpred/repo')
    !python -m scripts.train.train_tft \\
        --drive-dir /content/drive/MyDrive/stockpred_checkpoints

COLAB USAGE — notebook cell
----------------------------
    import importlib, scripts.train.train_tft as m
    importlib.reload(m)
    m.run(drive_dir="/content/drive/MyDrive/stockpred_checkpoints")

USAGE
-----
    python -m scripts.train.train_tft
    python -m scripts.train.train_tft --drive-dir /path/to/drive
    python -m scripts.train.train_tft --resume
    python -m scripts.train.train_tft --eval-only
"""

import argparse
import logging
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import yaml

# ── Project root: works both as a module (python -m) and in Colab cells ──────
try:
    PROJECT_ROOT = Path(__file__).resolve().parents[2]
except NameError:
    # Running inside a Colab/Jupyter cell — __file__ is not defined
    PROJECT_ROOT = Path("/content/drive/MyDrive/stockpred/repo")

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

# Hard-coded ASX 50 ticker list (order defines embedding index)
ASX_TICKERS = [
    "AFI.AX","AGL.AX","AMP.AX","ANZ.AX","ASX.AX","BEN.AX","BHP.AX","BOQ.AX",
    "BXB.AX","CBA.AX","CCP.AX","CGF.AX","CPU.AX","CSL.AX","DXS.AX","FMG.AX",
    "GPT.AX","IAG.AX","IEL.AX","IFL.AX","JBH.AX","LLC.AX","MFG.AX","MGR.AX",
    "MIN.AX","MQG.AX","NAB.AX","NCM.AX","NST.AX","ORG.AX","OZL.AX","QBE.AX",
    "QUB.AX","RHC.AX","RIO.AX","SCG.AX","SEK.AX","SHL.AX","STO.AX","SUN.AX",
    "SYD.AX","TAH.AX","TCL.AX","TLS.AX","TWE.AX","VCX.AX","WBC.AX","WES.AX",
    "WOW.AX","WPL.AX",
]
TICKER_TO_IDX = {t: i for i, t in enumerate(ASX_TICKERS)}
N_TICKERS = len(ASX_TICKERS)


# ─────────────────────────────────────────────────────────────────────────────
# TFT Building Blocks
# ─────────────────────────────────────────────────────────────────────────────

class GLU(nn.Module):
    """Gated Linear Unit: split activations, gate second half."""
    def __init__(self, d: int):
        super().__init__()
        self.linear = nn.Linear(d, d * 2)
        self.norm   = nn.LayerNorm(d)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h      = self.linear(x)
        h1, h2 = h.chunk(2, dim=-1)
        return self.norm(h1 * torch.sigmoid(h2))


class GRN(nn.Module):
    """
    Gated Residual Network.
    ELU(W1 * ELU(W2 * x + b2) + b1) with GLU + skip connection.
    """
    def __init__(self, d_in: int, d_hidden: int, d_out: int, dropout: float = 0.1):
        super().__init__()
        self.fc1  = nn.Linear(d_in, d_hidden)
        self.fc2  = nn.Linear(d_hidden, d_out)
        self.glu  = GLU(d_out)
        self.skip = nn.Linear(d_in, d_out) if d_in != d_out else nn.Identity()
        self.norm = nn.LayerNorm(d_out)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, context: torch.Tensor | None = None) -> torch.Tensor:
        h = nn.functional.elu(self.fc1(x))
        if context is not None:
            h = h + context
        h = self.drop(nn.functional.elu(self.fc2(h)))
        h = self.glu(h)
        return self.norm(h + self.skip(x))


class TFT(nn.Module):
    """
    Memory-efficient Temporal Fusion Transformer.

    Replaces the original VSN (260 per-feature GRNs × T timesteps) with a
    single Linear(n_features → d_model) + LayerNorm encoder. Everything else
    follows the TFT paper: GRN post-encoder, LSTM, multi-head self-attention,
    residual + LayerNorm, final GRN, dual regression + classification heads.

    Static ticker context is injected via Embedding → GRN → broadcast-add
    before the LSTM, matching the paper's static covariate enrichment step.

    Args:
        n_features:   number of input features per timestep (260)
        n_tickers:    vocabulary size for ticker embedding (50)
        d_model:      hidden dimension throughout the network
        lstm_layers:  number of stacked LSTM layers
        n_heads:      attention heads (must divide d_model evenly)
        dropout:      dropout probability
    """
    def __init__(
        self,
        n_features:  int,
        n_tickers:   int,
        d_model:     int   = 64,
        lstm_layers: int   = 1,
        n_heads:     int   = 4,
        dropout:     float = 0.1,
    ):
        super().__init__()
        self.d_model = d_model

        # ── Feature encoder (replaces VSN) ───────────────────────────────
        # A single linear projection is orders of magnitude cheaper than
        # 260 separate GRNs and avoids T4 OOM at any reasonable batch size.
        self.feat_encoder = nn.Sequential(
            nn.Linear(n_features, d_model),
            nn.LayerNorm(d_model),
            nn.Dropout(dropout),
        )

        # Post-encoder GRN (keeps non-linear feature mixing from TFT)
        self.post_enc_grn = GRN(d_model, d_model, d_model, dropout)

        # ── Static ticker context ─────────────────────────────────────────
        self.ticker_emb = nn.Embedding(n_tickers + 1, 16, padding_idx=0)
        self.static_grn = GRN(16, d_model, d_model, dropout)

        # ── LSTM encoder ──────────────────────────────────────────────────
        self.lstm = nn.LSTM(
            input_size  = d_model,
            hidden_size = d_model,
            num_layers  = lstm_layers,
            batch_first = True,
            dropout     = dropout if lstm_layers > 1 else 0.0,
        )

        # Post-LSTM GRN
        self.post_lstm_grn = GRN(d_model, d_model, d_model, dropout)

        # ── Multi-head self-attention ─────────────────────────────────────
        self.attn      = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )
        self.attn_norm = nn.LayerNorm(d_model)

        # ── Final GRN + output heads ──────────────────────────────────────
        self.final_grn = GRN(d_model, d_model, d_model, dropout)
        self.head_reg  = nn.Linear(d_model, 1)
        self.head_cls  = nn.Linear(d_model, 1)

        self.dropout   = nn.Dropout(dropout)

    def forward(
        self,
        x:          torch.Tensor,               # (B, L, F)
        ticker_idx: torch.Tensor,               # (B,)  int ticker indices
        pad_mask:   torch.Tensor | None = None, # (B, L) True = valid token
    ) -> tuple[torch.Tensor, torch.Tensor]:

        # ── Encode features ───────────────────────────────────────────────
        enc = self.feat_encoder(x)           # (B, L, d_model)
        enc = self.post_enc_grn(enc)         # (B, L, d_model)

        # ── Static ticker context ─────────────────────────────────────────
        static_emb = self.ticker_emb(ticker_idx)     # (B, 16)
        static_ctx = self.static_grn(static_emb)     # (B, d_model)
        enc = enc + static_ctx.unsqueeze(1)          # broadcast add
        enc = self.dropout(enc)

        # ── LSTM ──────────────────────────────────────────────────────────
        lstm_out, _ = self.lstm(enc)                 # (B, L, d_model)
        lstm_out    = self.post_lstm_grn(lstm_out)   # (B, L, d_model)

        # ── Multi-head attention ───────────────────────────────────────────
        attn_key_mask = None
        if pad_mask is not None:
            attn_key_mask = ~pad_mask   # True = padding (ignored by nn.MHA)

        attn_out, _ = self.attn(
            lstm_out, lstm_out, lstm_out,
            key_padding_mask=attn_key_mask,
        )
        attn_out = self.attn_norm(attn_out + lstm_out)  # residual

        # ── Final GRN → last timestep → heads ────────────────────────────
        out      = self.final_grn(attn_out)    # (B, L, d_model)
        last     = out[:, -1, :]              # (B, d_model)

        pred_ret = self.head_reg(last)         # (B, 1)
        pred_dir = self.head_cls(last)         # (B, 1)
        return pred_ret, pred_dir


# ─────────────────────────────────────────────────────────────────────────────
# Ticker index helper
# ─────────────────────────────────────────────────────────────────────────────

def tickers_to_idx(tickers: list[str], device: torch.device) -> torch.Tensor:
    """Convert ticker strings to embedding indices. Unknown tickers → 0 (padding)."""
    idx = [TICKER_TO_IDX.get(t, 0) + 1 for t in tickers]  # +1 to reserve 0 for padding
    return torch.tensor(idx, dtype=torch.long, device=device)


# ─────────────────────────────────────────────────────────────────────────────
# Training loop
# ─────────────────────────────────────────────────────────────────────────────

def run_epoch(
    model:     TFT,
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
            x         = batch["x"].to(device)          # (B, L, F)
            pad_mask  = batch["pad_mask"].to(device)   # (B, L)
            y_ret     = batch["y_ret"].to(device)
            y_dir     = batch["y_dir"].to(device)
            tickers_b = batch["ticker"]

            ticker_idx = tickers_to_idx(tickers_b, device)

            pred_ret, pred_dir = model(x, ticker_idx, pad_mask)
            loss, loss_info    = loss_fn(pred_ret, pred_dir, y_ret, y_dir)

            if is_train:
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.1)
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
    log.info("train_tft.py  Phase 4  (VSN-free, lookback=21)")
    log.info("=" * 65)

    with open(CONFIG_PATH) as f:
        cfg = yaml.safe_load(f)

    if drive_dir:
        cfg["training"]["drive_checkpoint_dir"] = drive_dir

    model_cfg = cfg["model_configs"]["tft"]
    train_cfg = cfg["training"]
    loss_cfg  = cfg["loss"]
    eval_cfg  = cfg["evaluation"]

    device = (torch.device("cuda") if torch.cuda.is_available()
              else torch.device("mps") if torch.backends.mps.is_available()
              else torch.device("cpu"))
    log.info(f"Device: {device}" +
             (f" ({torch.cuda.get_device_name(0)})" if device.type == "cuda" else ""))

    log.info("Building data loaders ...")
    train_loader, val_loader, test_loader = build_loaders(
        cfg, mode="sequence",
        num_workers=0,                          # avoid Colab shared-memory errors
        pin_memory=(device.type == "cuda"),
    )
    features   = load_locked_features(cfg["data"]["locked_config"])
    n_features = len(features)

    model = TFT(
        n_features  = n_features,
        n_tickers   = N_TICKERS,
        d_model     = model_cfg["hidden_size"],
        lstm_layers = model_cfg["lstm_layers"],
        n_heads     = model_cfg["attention_heads"],
        dropout     = model_cfg["dropout"],
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log.info(f"TFT (VSN-free): {n_params:,} parameters  features={n_features}  "
             f"d_model={model_cfg['hidden_size']}  "
             f"lookback={cfg['sequence']['lookback']}")

    loss_fn   = MultiTaskLoss(loss_cfg["reg_weight"], loss_cfg["cls_weight"])
    optimizer = optim.AdamW(
        model.parameters(),
        lr=model_cfg["learning_rate"],
        weight_decay=model_cfg["weight_decay"],
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", patience=10, factor=0.5, min_lr=1e-6
    )

    start_epoch, best_val_ic = 0, -999.0
    if resume or eval_only:
        model, optimizer, start_epoch, best_val_ic = load_checkpoint(
            model, optimizer, "tft", cfg, load_best=eval_only
        )

    if not eval_only:
        patience_counter = 0
        max_epochs       = model_cfg["max_epochs"]
        early_stopping   = train_cfg["early_stopping"]

        log.info(f"Training for up to {max_epochs} epochs ...")
        for epoch in range(start_epoch, max_epochs):
            t0 = time.time()

            train_loss, *_ = run_epoch(
                model, train_loader, loss_fn, optimizer, device, is_train=True
            )

            val_loss, pred_ret_v, pred_dir_v, y_ret_v, y_dir_v, dates_v, tickers_v = run_epoch(
                model, val_loader, loss_fn, None, device, is_train=False
            )
            val_eval = evaluate_predictions(
                pred_ret_v, pred_dir_v, y_ret_v, y_dir_v, dates_v, tickers_v,
                top_k_pct=eval_cfg["top_k_pct"], txn_cost=eval_cfg["transaction_cost"],
            )

            scheduler.step(val_eval.ic_mean)

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
                f"ICIR={val_eval.icir:+.3f}  HitRate={val_eval.hit_rate:.1%}  "
                f"L/S_Sharpe={val_eval.ls_sharpe:+.3f}  "
                f"best={best_val_ic:+.4f}  "
                f"patience={patience_counter}/{early_stopping}  "
                f"lr={lr_now:.2e}  [{time.time()-t0:.1f}s]"
            )

            save_checkpoint(model, optimizer, epoch, val_eval.ic_mean,
                            cfg, "tft", is_best)

            if patience_counter >= early_stopping:
                log.info(f"Early stopping at epoch {epoch+1}")
                break

    # ── Final evaluation ───────────────────────────────────────────────────
    log.info("Loading best checkpoint for final evaluation ...")
    model, _, _, _ = load_checkpoint(model, None, "tft", cfg, load_best=True)
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
    log.info("TFT TRAINING COMPLETE")
    log.info("=" * 65)


# ── Entry point ───────────────────────────────────────────────────────────────
# Colab-safe: argparse is only invoked when run as a script (python -m or
# python train_tft.py), not when exec'd or imported inside a notebook cell.
# parse_known_args silently ignores Colab's injected -f kernel.json argument.
# To call directly from a notebook cell use: run(drive_dir="...")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--drive-dir", type=str, default=None)
    parser.add_argument("--resume",    action="store_true")
    parser.add_argument("--eval-only", action="store_true")
    args, _ = parser.parse_known_args()   # _ silently discards -f kernel.json
    run(drive_dir=args.drive_dir, resume=args.resume, eval_only=args.eval_only)