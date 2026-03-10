"""
dataset.py
==========
Phase 4 — Shared dataset and DataLoader for all four models.

Provides two dataset classes:
  TabularDataset   — flat feature vector per row (LightGBM, FT-Transformer)
  SequenceDataset  — rolling lookback window per row (TFT, DLinear)

SCALING NOTE
------------
Assumes panel_train_scaled.parquet is pre-scaled on the source machine.
No runtime scaling is applied. Val and test use panel_*_scaled.parquet.
All three splits are loaded identically — just read and go.

NaN HANDLING
------------
NaN values in features are replaced with 0.0 in tensors and indicated
via a boolean mask (True = valid). LightGBM handles NaN natively and
does not use these datasets — it reads numpy arrays directly.
"""

import json
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def load_locked_features(locked_path: str) -> list:
    with open(locked_path) as f:
        return json.load(f)["features"]


def load_dataframe(path: str, features: list) -> pd.DataFrame:
    """Load parquet, parse dates, sort by ticker+date."""
    df = pd.read_parquet(path)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["ticker", "date"]).reset_index(drop=True)
    # Verify all features present
    missing = [f for f in features if f not in df.columns]
    if missing:
        raise ValueError(f"Missing {len(missing)} features in {path}: {missing[:5]}")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Tabular Dataset (FT-Transformer)
# ─────────────────────────────────────────────────────────────────────────────

class TabularDataset(Dataset):
    """
    Flat feature vector per prediction row.
    Returns zero-imputed float32 tensors with NaN mask.
    """

    def __init__(self, df: pd.DataFrame, features: list):
        self.features = features
        self.tickers  = df["ticker"].values
        self.dates    = df["date"].astype(str).values
        self.y_ret    = df["y_ret_21d"].astype("float32").values
        self.y_dir    = df["y_dir_21d"].astype("float32").values

        feat_np       = df[features].astype("float32").values
        self.nan_mask = np.isnan(feat_np)
        self.X        = np.nan_to_num(feat_np, nan=0.0)

        log.info(
            f"TabularDataset: {len(df):,} rows  {len(features)} features  "
            f"nan_rate={self.nan_mask.mean():.2%}"
        )

    def __len__(self):
        return len(self.y_ret)

    def __getitem__(self, idx):
        return {
            "x":      torch.from_numpy(self.X[idx]),
            "mask":   torch.from_numpy(~self.nan_mask[idx]),
            "y_ret":  torch.tensor(self.y_ret[idx], dtype=torch.float32),
            "y_dir":  torch.tensor(self.y_dir[idx], dtype=torch.float32),
            "ticker": self.tickers[idx],
            "date":   self.dates[idx],
        }


# ─────────────────────────────────────────────────────────────────────────────
# Sequence Dataset (TFT, DLinear)
# ─────────────────────────────────────────────────────────────────────────────

class SequenceDataset(Dataset):
    """
    Rolling lookback window per prediction.

    For each valid (date, ticker) target row, constructs a sequence of
    the previous `lookback` days of features for that ticker.
    Left-pads with zeros when history < lookback.
    """

    def __init__(self, df: pd.DataFrame, features: list, lookback: int = 126):
        self.features   = features
        self.lookback   = lookback
        self.n_features = len(features)

        self.tickers_all = df["ticker"].values
        self.dates_all   = df["date"].astype(str).values
        self.y_ret_all   = df["y_ret_21d"].astype("float32").values
        self.y_dir_all   = df["y_dir_21d"].astype("float32").values

        feat_np       = df[features].astype("float32").values
        self.feat_np  = np.nan_to_num(feat_np, nan=0.0)

        # Build valid index: (global_row_idx, seq_indices_array)
        self.valid_indices = []
        ticker_groups = df.groupby("ticker", observed=True, sort=False).indices

        for ticker, indices in ticker_groups.items():
            indices = sorted(indices)
            for pos, global_idx in enumerate(indices):
                if not np.isnan(self.y_ret_all[global_idx]):
                    seq_start = max(0, pos - lookback + 1)
                    self.valid_indices.append(
                        (global_idx, indices[seq_start: pos + 1])
                    )

        log.info(
            f"SequenceDataset: {len(self.valid_indices):,} valid samples  "
            f"lookback={lookback}  features={self.n_features}"
        )

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):
        global_idx, seq_indices = self.valid_indices[idx]
        seq_len  = len(seq_indices)
        seq_data = self.feat_np[seq_indices]             # (seq_len, F)

        if seq_len < self.lookback:
            pad_len  = self.lookback - seq_len
            padding  = np.zeros((pad_len, self.n_features), dtype=np.float32)
            seq_data = np.concatenate([padding, seq_data], axis=0)
            pad_mask = np.array([False] * pad_len + [True] * seq_len, dtype=bool)
        else:
            pad_mask = np.ones(self.lookback, dtype=bool)

        return {
            "x":        torch.from_numpy(seq_data),
            "pad_mask": torch.from_numpy(pad_mask),
            "y_ret":    torch.tensor(self.y_ret_all[global_idx], dtype=torch.float32),
            "y_dir":    torch.tensor(self.y_dir_all[global_idx], dtype=torch.float32),
            "ticker":   self.tickers_all[global_idx],
            "date":     self.dates_all[global_idx],
        }


# ─────────────────────────────────────────────────────────────────────────────
# DataLoader builder
# ─────────────────────────────────────────────────────────────────────────────

def build_loaders(
    cfg:         dict,
    mode:        str = "tabular",
    num_workers: int = 2,
    pin_memory:  bool = True,
):
    """
    Build train/val/test DataLoaders.

    Args:
        cfg:         train_config dict
        mode:        "tabular" for FT-Transformer, "sequence" for TFT/DLinear
        num_workers: DataLoader workers (use 0 on Windows or if errors)
        pin_memory:  True for GPU training

    Returns:
        (train_loader, val_loader, test_loader)
    """
    data_cfg  = cfg["data"]
    features  = load_locked_features(data_cfg["locked_config"])
    lookback  = cfg["sequence"]["lookback"]

    log.info(f"Loading train split ({mode}) ...")
    train_df = load_dataframe(data_cfg["train_path"], features)

    log.info("Loading val split ...")
    val_df = load_dataframe(data_cfg["val_path"], features)

    log.info("Loading test split ...")
    test_df = load_dataframe(data_cfg["test_path"], features)

    if mode == "tabular":
        train_ds = TabularDataset(train_df, features)
        val_ds   = TabularDataset(val_df,   features)
        test_ds  = TabularDataset(test_df,  features)
        batch_size = cfg["model_configs"]["ft_transformer"]["batch_size"]

    elif mode == "sequence":
        train_ds = SequenceDataset(train_df, features, lookback=lookback)
        val_ds   = SequenceDataset(val_df,   features, lookback=lookback)
        test_ds  = SequenceDataset(test_df,  features, lookback=lookback)
        batch_size = 256

    else:
        raise ValueError(f"mode must be 'tabular' or 'sequence', got: {mode}")

    pin = pin_memory and torch.cuda.is_available()

    def make_loader(ds, shuffle):
        return DataLoader(
            ds, batch_size=batch_size, shuffle=shuffle,
            num_workers=num_workers, pin_memory=pin, drop_last=False,
        )

    return (
        make_loader(train_ds, shuffle=True),
        make_loader(val_ds,   shuffle=False),
        make_loader(test_ds,  shuffle=False),
    )
