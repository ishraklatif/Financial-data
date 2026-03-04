#!/usr/bin/env python3
"""
build_feature_panel.py
======================
Phase 3 — Master feature engineering script.

Reads all 120 raw parquets, aligns to ASX trading calendar, computes
all derived features, and writes a single stacked panel parquet.

TARGET VARIABLE
---------------
y_ret_21d   : 21-trading-day forward log return (close_adj)
y_dir_21d   : 1 if y_ret_21d > 0 else 0 (binary direction label)

Both are computed here but NOT included in the feature matrix —
they are written as separate columns so compute_targets.py can
quarantine and validate them before training.

LEAKAGE DISCIPLINE
------------------
- NO ffill is applied here. All NaN values from alignment are left
  as NaN. ffill is applied ONLY inside each split in split_and_fill.py.
- Macro series are aligned two ways:
    <name>        : forward-filled from period-end date (optimistic)
    <name>_lag    : forward-filled from release date   (strict, no look-ahead)
  The release lag offsets are hardcoded below and documented.
- Targets use shift(-21) on the SPINE date index, so they always
  refer to 21 trading days ahead from each row's date.

OUTPUT
------
data/features/panel_raw.parquet
  Columns: date, ticker, [all features], y_ret_21d, y_dir_21d
  Rows: ASX trading days × tickers (only from each ticker's list_date)
  ~267,000 rows × ~210 columns

data/features/panel_build_log.json
  Build metadata, feature counts, coverage stats per ticker

USAGE
-----
    python -m scripts.build.build_feature_panel
    python -m scripts.build.build_feature_panel --verify   # check output only
    python -m scripts.build.build_feature_panel --ticker CBA.AX  # single ticker
"""

import argparse
import json
import logging
import sys
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(PROJECT_ROOT / "logs" / "build_feature_panel.log"),
    ],
)
log = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────────────────────────────────────

RAW          = PROJECT_ROOT / "data" / "raw"
FEATURES_DIR = PROJECT_ROOT / "data" / "features"
CONFIG_DIR   = PROJECT_ROOT / "config"

CALENDAR_PATH   = RAW / "calendar" / "ASX_CALENDAR.parquet"
UNIVERSE_PATH   = CONFIG_DIR / "universe_manifest.json"
DATA_YAML_PATH  = CONFIG_DIR / "data.yaml"

# ─────────────────────────────────────────────────────────────────────────────
# Release lag offsets for strict (_lag) alignment
# Source: typical publication delays for each series
# ─────────────────────────────────────────────────────────────────────────────

# Format: canonical_name → business days after period-end before data is public
RELEASE_LAGS = {
    # RBA macro (ABS quarterly) — released ~60 days after quarter end
    "CPI_INDEX":  63,
    "CPI_YOY":    63,
    "UNEMP":      35,   # monthly, ABS Labour Force ~5 weeks after reference month
    # ABS quarterly — released ~65 days after quarter end
    "GDP":        65,
    "GDP_PCA":    65,
    "HSR":        65,
    "TOT":        65,
    "WPI":        55,   # quarterly, slightly faster
    # FRED monthly — typically released 3–5 weeks after reference month
    "CPI":        30,
    "PCEPI":      30,
    "FEDFUNDS":   30,
    "INDPRO":     30,
    "PMI_MFG":    30,
    "PMI_SERV":   30,
    "JOLTS":      45,   # JOLTS has extra lag
    "UMICH":      25,
    # FRED quarterly
    "GDP_FRED":   65,   # US GDP
    # F3 credit yields — monthly, RBA publishes ~6 weeks after month end
    "F3_A_YIELD_3Y":    40,
    "F3_A_YIELD_5Y":    40,
    "F3_A_YIELD_7Y":    40,
    "F3_A_YIELD_10Y":   40,
    "F3_BBB_YIELD_3Y":  40,
    "F3_BBB_YIELD_5Y":  40,
    "F3_BBB_YIELD_7Y":  40,
    "F3_BBB_YIELD_10Y": 40,
}

# ─────────────────────────────────────────────────────────────────────────────
# Load helpers
# ─────────────────────────────────────────────────────────────────────────────

def load_config() -> dict:
    with open(DATA_YAML_PATH) as f:
        return yaml.safe_load(f)


def load_universe() -> dict:
    with open(UNIVERSE_PATH) as f:
        return json.load(f)


def load_calendar() -> pd.DatetimeIndex:
    cal = pd.read_parquet(CALENDAR_PATH)
    cal["date"] = pd.to_datetime(cal["date"])
    trading = cal[cal["is_trading_day"] == True]["date"]
    return pd.DatetimeIndex(sorted(trading))


def load_parquet(path: Path, date_col: str = "date") -> pd.DataFrame:
    df = pd.read_parquet(path)
    df[date_col] = pd.to_datetime(df[date_col])
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Alignment utilities
# ─────────────────────────────────────────────────────────────────────────────

def align_daily(df: pd.DataFrame, spine: pd.DatetimeIndex,
                value_col: str = "value", name: str = "") -> pd.Series:
    """
    Align a daily series to the ASX spine.
    No ffill — just reindex. NaN on days the source has no observation.
    (US/UK market holidays, etc.)
    """
    df = df.set_index("date")[value_col].sort_index()
    return df.reindex(spine).rename(name or value_col)


def align_low_freq(df: pd.DataFrame, spine: pd.DatetimeIndex,
                   value_col: str = "value", name: str = "",
                   release_lag_days: int = 0) -> tuple[pd.Series, pd.Series]:
    """
    Align a monthly/quarterly series to the ASX spine.
    Returns two series:
      - optimistic: forward-fill from period-end date (no lag)
      - strict (_lag): forward-fill from period-end + release_lag offset

    Both series are placed on the spine but NOT filled here —
    the caller receives sparse series. ffill happens in split_and_fill.py.

    The lag is applied by shifting observation dates forward by
    release_lag_days calendar days before reindexing.
    """
    s = df.set_index("date")[value_col].sort_index()

    # Optimistic: value available on period-end date
    s_opt = s.reindex(spine).rename(name)

    # Strict: value only available after release lag
    if release_lag_days > 0:
        lag_index = s.index + pd.offsets.BusinessDay(release_lag_days)
        s_lag = pd.Series(s.values, index=lag_index)
        # Remove any duplicates from BDay rounding
        s_lag = s_lag[~s_lag.index.duplicated(keep="last")]
        s_lag = s_lag.reindex(spine).rename(f"{name}_lag")
    else:
        s_lag = s.reindex(spine).rename(f"{name}_lag")

    return s_opt, s_lag


def align_ohlcv(df: pd.DataFrame, spine: pd.DatetimeIndex,
                prefix: str = "") -> pd.DataFrame:
    """
    Align an OHLCV parquet (indices, FX, commodities, sector ETFs)
    to the ASX spine. Returns close only (and close_adj if present).
    """
    df = df.set_index("date").sort_index()
    cols = {}

    close_col = "close_adj" if "close_adj" in df.columns else "close"
    cols[f"{prefix}close"] = df[close_col].reindex(spine)

    if "volume" in df.columns:
        cols[f"{prefix}volume"] = df["volume"].reindex(spine)

    return pd.DataFrame(cols, index=spine)


# ─────────────────────────────────────────────────────────────────────────────
# Feature computation
# ─────────────────────────────────────────────────────────────────────────────

def log_return(s: pd.Series) -> pd.Series:
    """Log return: ln(p_t / p_{t-1})"""
    return np.log(s / s.shift(1))


def compute_equity_features(df: pd.DataFrame, spine: pd.DatetimeIndex,
                             ticker: str) -> pd.DataFrame:
    """
    Compute all equity-level features for one ticker.
    Input: company OHLCV parquet aligned to spine.
    """
    df = df.set_index("date").reindex(spine).sort_index()

    price = df["close_adj"]
    vol   = df["volume"]

    features = {}

    # ── Price features ────────────────────────────────────────────────────
    features["eq_close_adj"]   = price
    features["eq_log_ret_1d"]  = log_return(price)
    features["eq_log_ret_5d"]  = np.log(price / price.shift(5))
    features["eq_log_ret_21d"] = np.log(price / price.shift(21))

    # ── Momentum ─────────────────────────────────────────────────────────
    features["eq_mom_21d"]  = price / price.shift(21) - 1
    features["eq_mom_63d"]  = price / price.shift(63) - 1
    features["eq_mom_126d"] = price / price.shift(126) - 1
    features["eq_mom_252d"] = price / price.shift(252) - 1

    # ── Realised volatility ───────────────────────────────────────────────
    ret = features["eq_log_ret_1d"]
    features["eq_rvol_5d"]  = ret.rolling(5,  min_periods=3).std()
    features["eq_rvol_21d"] = ret.rolling(21, min_periods=10).std()
    features["eq_rvol_63d"] = ret.rolling(63, min_periods=30).std()

    # ── Volume features ───────────────────────────────────────────────────
    if vol.notna().sum() > 100:
        features["eq_log_volume"]     = np.log1p(vol)
        features["eq_volume_ma21_ratio"] = vol / vol.rolling(21, min_periods=10).mean()
    else:
        features["eq_log_volume"]        = pd.Series(np.nan, index=spine)
        features["eq_volume_ma21_ratio"] = pd.Series(np.nan, index=spine)

    # ── OHLC spread features ──────────────────────────────────────────────
    if "high" in df.columns and "low" in df.columns:
        hl_range = (df["high"] - df["low"]) / df["close_adj"]
        features["eq_hl_range"] = hl_range
        # Parkinson estimator of daily volatility
        features["eq_parkinson_vol"] = np.sqrt(
            (1 / (4 * np.log(2))) * (np.log(df["high"] / df["low"]) ** 2)
        )
    else:
        features["eq_hl_range"]      = pd.Series(np.nan, index=spine)
        features["eq_parkinson_vol"] = pd.Series(np.nan, index=spine)

    # ── Moving average ratios ─────────────────────────────────────────────
    features["eq_ma5_ratio"]  = price / price.rolling(5,   min_periods=3).mean()
    features["eq_ma21_ratio"] = price / price.rolling(21,  min_periods=10).mean()
    features["eq_ma63_ratio"] = price / price.rolling(63,  min_periods=30).mean()

    # ── Drawdown ──────────────────────────────────────────────────────────
    rolling_max = price.rolling(252, min_periods=63).max()
    features["eq_drawdown_252d"] = price / rolling_max - 1

    result = pd.DataFrame(features, index=spine)
    result.index.name = "date"
    return result


def compute_targets(price: pd.Series, spine: pd.DatetimeIndex) -> pd.DataFrame:
    """
    Compute 21-trading-day forward return and direction label.
    Uses shift(-21) on the spine — looks exactly 21 rows ahead.
    These are NOT features. They are the prediction targets.
    Quarantine and validation happens in compute_targets.py.
    """
    log_ret_21d_fwd = np.log(price.shift(-21) / price)
    direction       = (log_ret_21d_fwd > 0).astype(float)
    direction[log_ret_21d_fwd.isna()] = np.nan

    return pd.DataFrame({
        "y_ret_21d": log_ret_21d_fwd,
        "y_dir_21d": direction,
    }, index=spine)


def compute_index_features(close: pd.Series, name: str) -> pd.DataFrame:
    """Derived features for an index/ETF/FX/commodity close series."""
    features = {}
    features[f"{name}_close"]      = close
    features[f"{name}_ret_1d"]     = log_return(close)
    features[f"{name}_ret_5d"]     = np.log(close / close.shift(5))
    features[f"{name}_ret_21d"]    = np.log(close / close.shift(21))
    features[f"{name}_rvol_21d"]   = log_return(close).rolling(21, min_periods=10).std()
    features[f"{name}_ma21_ratio"] = close / close.rolling(21, min_periods=10).mean()
    return pd.DataFrame(features, index=close.index)


def compute_yield_curve_features(dgs2: pd.Series, dgs10: pd.Series,
                                  dgs30: pd.Series, dgs3mo: pd.Series,
                                  cash_rate: pd.Series) -> pd.DataFrame:
    """Yield curve spread and slope features."""
    features = {}
    # US yield curve spreads
    features["yc_us_10y_2y"]    = dgs10 - dgs2
    features["yc_us_30y_10y"]   = dgs30 - dgs10
    features["yc_us_10y_3mo"]   = dgs10 - dgs3mo
    features["yc_us_level_2y"]  = dgs2
    features["yc_us_level_10y"] = dgs10
    # AUS yield curve (using CASH_RATE as short end proxy pre-2013)
    features["yc_aus_cash_rate"] = cash_rate
    # Level and change
    features["yc_us_2y_ch21d"]  = dgs2  - dgs2.shift(21)
    features["yc_us_10y_ch21d"] = dgs10 - dgs10.shift(21)
    return pd.DataFrame(features)


def compute_credit_features(hy: pd.Series, ig: pd.Series,
                             ted: pd.Series) -> pd.DataFrame:
    """Credit spread features."""
    features = {}
    features["cr_hy_spread"]      = hy
    features["cr_ig_spread"]      = ig
    features["cr_hy_ig_ratio"]    = hy / ig.replace(0, np.nan)
    features["cr_ted_spread"]     = ted
    features["cr_hy_ch21d"]       = hy - hy.shift(21)
    features["cr_ig_ch21d"]       = ig - ig.shift(21)
    return pd.DataFrame(features)


def compute_short_interest_features(si_df: pd.DataFrame,
                                     ticker: str,
                                     spine: pd.DatetimeIndex) -> pd.DataFrame:
    """Short interest % for a specific ticker aligned to spine."""
    ticker_code = ticker.replace(".AX", "").replace(".", "")
    sub = si_df[si_df["product_code"] == ticker_code].copy()
    if len(sub) == 0:
        return pd.DataFrame({"si_short_pct":    pd.Series(np.nan, index=spine),
                              "si_short_ch21d":  pd.Series(np.nan, index=spine)},
                             index=spine)
    sub = sub.set_index("date")["short_pct"].sort_index()
    s = sub.reindex(spine)
    return pd.DataFrame({
        "si_short_pct":   s,
        "si_short_ch21d": s - s.shift(21),
    }, index=spine)


def compute_sector_relative(ticker_ret: pd.Series,
                              sector_rets: dict[str, pd.Series]) -> pd.DataFrame:
    """
    Relative return of ticker vs its GICS sector proxy ETF.
    sector_rets: dict of {etf_name: log_return_series}
    """
    features = {}
    for etf, ret in sector_rets.items():
        diff = ticker_ret - ret
        features[f"sr_vs_{etf.lower()}_1d"]  = diff
        features[f"sr_vs_{etf.lower()}_21d"] = (
            ticker_ret.rolling(21, min_periods=10).sum()
            - ret.rolling(21, min_periods=10).sum()
        )
    return pd.DataFrame(features, index=ticker_ret.index)


# ─────────────────────────────────────────────────────────────────────────────
# Main build
# ─────────────────────────────────────────────────────────────────────────────

def build(tickers_filter: list[str] | None = None) -> None:
    cfg      = load_config()
    universe = load_universe()
    spine    = load_calendar()
    start    = pd.Timestamp(cfg["data"]["start_date"])

    log.info("=" * 65)
    log.info("build_feature_panel.py — Phase 3 feature engineering")
    log.info(f"Spine: {spine[0].date()} → {spine[-1].date()} ({len(spine)} trading days)")
    log.info("=" * 65)

    FEATURES_DIR.mkdir(parents=True, exist_ok=True)

    # ── Load all shared (non-ticker-specific) series once ─────────────────

    log.info("Loading shared macro/market series ...")

    # Calendar features
    cal_df = pd.read_parquet(CALENDAR_PATH)
    cal_df["date"] = pd.to_datetime(cal_df["date"])
    cal_df = cal_df[cal_df["is_trading_day"] == True].set_index("date")
    cal_cols = [c for c in cal_df.columns if c.startswith("cal_") or c == "td_idx"]

    # AXJO index
    axjo = load_parquet(RAW / "macro" / "indices" / "AXJO.parquet")
    axjo_close = axjo.set_index("date")["close_adj"].reindex(spine)
    axjo_feats = compute_index_features(axjo_close, "xa_axjo")

    # Global indices
    idx_map = {
        "xa_gspc":  RAW / "macro" / "indices" / "GSPC.parquet",
        "xa_ftse":  RAW / "macro" / "indices" / "FTSE.parquet",
        "xa_n225":  RAW / "macro" / "indices" / "N225.parquet",
        "xa_hsi":   RAW / "macro" / "indices" / "HSI.parquet",
        "xa_sse":   RAW / "macro" / "indices" / "SSE.parquet",
        "xa_csi300":RAW / "macro" / "indices" / "CSI300.parquet",
    }
    idx_feats = {}
    for name, path in idx_map.items():
        if path.exists():
            df = load_parquet(path)
            close = df.set_index("date")["close_adj" if "close_adj" in df.columns else "close"].reindex(spine)
            idx_feats[name] = compute_index_features(close, name)

    # Volatility
    vix  = load_parquet(RAW / "macro" / "volatility" / "VIX.parquet")
    vvix = load_parquet(RAW / "macro" / "volatility" / "VVIX.parquet")
    move = load_parquet(RAW / "macro" / "volatility" / "MOVE.parquet")
    vol_feats = pd.DataFrame({
        "xa_vix":          vix.set_index("date")["close"].reindex(spine),
        "xa_vvix":         vvix.set_index("date")["close"].reindex(spine),
        "xa_move":         move.set_index("date")["close"].reindex(spine),
        "xa_vix_ret_1d":   log_return(vix.set_index("date")["close"].reindex(spine)),
        "xa_vix_ma21":     vix.set_index("date")["close"].reindex(spine).rolling(21, min_periods=10).mean(),
        "xa_vix_vs_ma21":  vix.set_index("date")["close"].reindex(spine) /
                           vix.set_index("date")["close"].reindex(spine).rolling(21, min_periods=10).mean(),
    }, index=spine)

    # FX
    fx_map = {
        "xa_audusd": RAW / "market" / "fx" / "AUDUSD.parquet",
        "xa_audjpy": RAW / "market" / "fx" / "AUDJPY.parquet",
        "xa_audcny": RAW / "market" / "fx" / "AUDCNY.parquet",
        "xa_dxy":    RAW / "market" / "fx" / "DXY.parquet",
    }
    fx_feats = {}
    for name, path in fx_map.items():
        if path.exists():
            df = load_parquet(path)
            close = df.set_index("date")["close" if "close" in df.columns else "close_adj"].reindex(spine)
            fx_feats[name] = compute_index_features(close, name)

    # Commodities
    comm_map = {
        "xa_gold":   RAW / "market" / "commodities" / "GOLD.parquet",
        "xa_oil":    RAW / "market" / "commodities" / "OIL.parquet",
        "xa_copper": RAW / "market" / "commodities" / "COPPER.parquet",
        "xa_silver": RAW / "market" / "commodities" / "SILVER.parquet",
        "xa_iron":   RAW / "market" / "commodities" / "IRON.parquet",
        "xa_dbc":    RAW / "market" / "commodities" / "DBC.parquet",
    }
    comm_feats = {}
    for name, path in comm_map.items():
        if path.exists():
            df = load_parquet(path)
            close_col = "close_adj" if "close_adj" in df.columns else "close"
            close = df.set_index("date")[close_col].reindex(spine)
            comm_feats[name] = compute_index_features(close, name)

    # Sector ETFs (US proxies)
    # NOTE on calendar alignment: US and ASX trade on ~50% different days.
    # For sector FEATURES (xs_*): keep sparse — late-start ETFs (XLRE, XLC)
    #   should remain NaN pre-launch so lock_features.py drops them correctly.
    # For sector RETURNS used in relative-performance (sr_*): ffill prices to
    #   ASX spine before computing log_return. On ASX days when US was closed,
    #   price is unchanged → return = 0, which is correct (no new US information).
    sector_etfs = ["XLF","XLK","XLI","XLE","XLV","XLU","XLB","XLP","XLY","XLRE","XLC","GDX"]
    sector_rets = {}
    sector_feats = {}
    for etf in sector_etfs:
        path = RAW / "sector" / f"{etf}.parquet"
        if path.exists():
            df = load_parquet(path)
            close_col = "close_adj" if "close_adj" in df.columns else "close"
            close = df.set_index("date")[close_col].reindex(spine)
            # For xs_ features: use sparse close (preserves late-start NaN gaps)
            sector_feats[etf] = compute_index_features(close, f"xs_{etf.lower()}")
            # For sr_ relative performance: ffill to ASX calendar so returns are
            # defined on every ASX trading day (0 on days US was closed)
            close_for_ret = close.ffill()
            ret = log_return(close_for_ret)
            ret.index.name = "date"  # match ticker_ret index name to avoid NaN on subtraction
            sector_rets[etf] = ret

    # FRED rates
    dgs2   = load_parquet(RAW / "fred" / "DGS2.parquet").set_index("date")["value"].reindex(spine)
    dgs10  = load_parquet(RAW / "fred" / "DGS10.parquet").set_index("date")["value"].reindex(spine)
    dgs30  = load_parquet(RAW / "fred" / "DGS30.parquet").set_index("date")["value"].reindex(spine)
    dgs3mo = load_parquet(RAW / "fred" / "DGS3MO.parquet").set_index("date")["value"].reindex(spine)

    # Credit spreads
    hy  = load_parquet(RAW / "fred" / "BAMLH0A0HYM2.parquet").set_index("date")["value"].reindex(spine)
    ig  = load_parquet(RAW / "fred" / "BAMLCC0A1AAATRIV.parquet").set_index("date")["value"].reindex(spine)
    # Use TED_SPREAD if available, else fall back to TEDRATE
    ted_path = RAW / "fred" / "TED_SPREAD.parquet"
    if not ted_path.exists():
        ted_path = RAW / "fred" / "TEDRATE.parquet"
    ted = load_parquet(ted_path).set_index("date")["value"].reindex(spine)

    # RBA rates
    cash_rate = load_parquet(RAW / "rba" / "rates" / "CASH_RATE.parquet").set_index("date")["value"].reindex(spine)
    yield_2y_path  = RAW / "rba" / "rates" / "YIELD_2Y.parquet"
    yield_10y_path = RAW / "rba" / "rates" / "YIELD_10Y.parquet"
    yield_2y  = load_parquet(yield_2y_path).set_index("date")["value"].reindex(spine)  if yield_2y_path.exists()  else pd.Series(np.nan, index=spine)
    yield_10y = load_parquet(yield_10y_path).set_index("date")["value"].reindex(spine) if yield_10y_path.exists() else pd.Series(np.nan, index=spine)

    yc_feats = compute_yield_curve_features(dgs2, dgs10, dgs30, dgs3mo, cash_rate)
    yc_feats.index = spine
    # Add AUS-specific yield features
    yc_feats["yc_aus_2y"]      = yield_2y
    yc_feats["yc_aus_10y"]     = yield_10y
    yc_feats["yc_aus_10y_2y"]  = yield_10y - yield_2y
    yc_feats["yc_aus_10y_cash"]= yield_10y - cash_rate
    cr_feats = compute_credit_features(hy, ig, ted)
    cr_feats.index = spine

    # Low-frequency macro series (both optimistic and strict lag)
    macro_low_freq_raw = {}

    def _load_lf(path: Path, name: str) -> None:
        if not path.exists():
            return
        df = load_parquet(path)
        opt, lag = align_low_freq(
            df, spine,
            value_col="value",
            name=name,
            release_lag_days=RELEASE_LAGS.get(name, 30)
        )
        macro_low_freq_raw[name]         = opt
        macro_low_freq_raw[f"{name}_lag"] = lag

    # AUS macro
    _load_lf(RAW / "rba"  / "macro" / "CPI_INDEX.parquet",  "CPI_INDEX")
    _load_lf(RAW / "rba"  / "macro" / "CPI_YOY.parquet",    "CPI_YOY")
    _load_lf(RAW / "rba"  / "macro" / "UNEMP.parquet",      "UNEMP_AUS")
    _load_lf(RAW / "abs"  / "gdp"   / "GDP.parquet",        "GDP_AUS")
    _load_lf(RAW / "abs"  / "gdp"   / "GDP_PCA.parquet",    "GDP_PCA_AUS")
    _load_lf(RAW / "abs"  / "gdp"   / "HSR.parquet",        "HSR")
    _load_lf(RAW / "abs"  / "gdp"   / "TOT.parquet",        "TOT")
    _load_lf(RAW / "abs"  / "wpi"   / "WPI.parquet",        "WPI")
    # US macro
    _load_lf(RAW / "fred" / "CPIAUCSL.parquet", "CPI_US")
    _load_lf(RAW / "fred" / "PCEPI.parquet",    "PCEPI")
    _load_lf(RAW / "fred" / "UNRATE.parquet",   "UNEMP_US")
    _load_lf(RAW / "fred" / "FEDFUNDS.parquet", "FEDFUNDS")
    _load_lf(RAW / "fred" / "GDPC1.parquet",    "GDP_US")
    _load_lf(RAW / "fred" / "INDPRO.parquet",   "INDPRO")
    _load_lf(RAW / "fred" / "IPMAN.parquet",    "PMI_MFG")
    _load_lf(RAW / "fred" / "SRVPRD.parquet",   "PMI_SERV")
    _load_lf(RAW / "fred" / "JTSJOL.parquet",   "JOLTS")
    _load_lf(RAW / "fred" / "UMCSENT.parquet",  "UMICH")
    # F3 corporate bond yields (monthly, RBA)
    for tenor in ["3Y", "5Y", "7Y", "10Y"]:
        for rating in ["A", "BBB"]:
            nm = f"F3_{rating}_YIELD_{tenor}"
            _load_lf(RAW / "rba" / "credit" / f"{nm}.parquet", nm)

    macro_lf_df = pd.DataFrame(macro_low_freq_raw, index=spine)

    # Short interest
    si_path = RAW / "sentiment" / "short_interest" / "SHORT_INTEREST.parquet"
    si_df = load_parquet(si_path) if si_path.exists() else pd.DataFrame()

    # ── Assemble shared block (same for every ticker) ─────────────────────
    shared_blocks = [
        cal_df[cal_cols],
        axjo_feats,
        vol_feats,
        yc_feats,
        cr_feats,
        macro_lf_df,
    ]
    for feats in idx_feats.values():
        shared_blocks.append(feats)
    for feats in fx_feats.values():
        shared_blocks.append(feats)
    for feats in comm_feats.values():
        shared_blocks.append(feats)
    for feats in sector_feats.values():
        shared_blocks.append(feats)

    shared = pd.concat(shared_blocks, axis=1)
    shared.index = spine

    log.info(f"Shared block: {shared.shape[1]} columns")

    # ── Determine tickers ─────────────────────────────────────────────────
    company_tickers = [
        k for k, v in universe["tickers"].items()
        if v.get("history_type", "equity") in ("equity", "demerger", "listing")
    ]
    if tickers_filter:
        company_tickers = [t for t in company_tickers if t in tickers_filter]

    log.info(f"Processing {len(company_tickers)} tickers ...")

    # ── Sector ETF returns for relative performance ───────────────────────
    # Map tickers to their GICS sector ETF
    GICS_MAP = {
        # Financials
        "CBA.AX":"XLF","NAB.AX":"XLF","WBC.AX":"XLF","ANZ.AX":"XLF",
        "MQG.AX":"XLF","SUN.AX":"XLF","IAG.AX":"XLF","AMP.AX":"XLF",
        "ASX.AX":"XLF","CPU.AX":"XLF","CCP.AX":"XLF",
        # Materials
        "BHP.AX":"XLB","RIO.AX":"XLB","FMG.AX":"XLB","S32.AX":"XLB",
        "ORI.AX":"XLB","NEM.AX":"GDX",
        # Energy
        "WDS.AX":"XLE","STO.AX":"XLE","ALD.AX":"XLE","ORG.AX":"XLE","AGL.AX":"XLU",
        # Industrials/Infrastructure
        "TCL.AX":"XLI","QAN.AX":"XLI","BXB.AX":"XLI","APA.AX":"XLU",
        "AIA.AX":"XLI","AZJ.AX":"XLI",
        # Real estate
        "GPT.AX":"XLRE","MGR.AX":"XLRE","SCG.AX":"XLRE","NSR.AX":"XLRE","LLC.AX":"XLRE",
        # Health
        "CSL.AX":"XLV","RHC.AX":"XLV","SHL.AX":"XLV",
        # Consumer discretionary
        "JBH.AX":"XLY","DMP.AX":"XLY","SUL.AX":"XLY","COL.AX":"XLP","MTS.AX":"XLP",
        "WES.AX":"XLY","WOW.AX":"XLP",
        # Consumer staples
        # Telecom / Comm services
        "TLS.AX":"XLC","TPG.AX":"XLC",
        # Tech / Comms
        "SEK.AX":"XLC","CAR.AX":"XLC","WTC.AX":"XLK","XRO.AX":"XLK",
        # Materials packaging
        "AMC.AX":"XLB","JHX.AX":"XLB",
    }

    # ── Per-ticker loop ───────────────────────────────────────────────────
    all_panels = []
    build_log  = []

    for ticker in company_tickers:
        info     = universe["tickers"].get(ticker, {})
        list_date = pd.Timestamp(info.get("list_date", str(start.date())))

        # Trim spine to ticker's list date
        ticker_spine = spine[spine >= list_date]
        if len(ticker_spine) < 63:
            log.warning(f"  {ticker}: only {len(ticker_spine)} trading days — skipping")
            continue

        # Load company OHLCV
        safe_name = ticker.replace(".", "_")
        co_path   = RAW / "companies" / f"{safe_name}.parquet"
        if not co_path.exists():
            log.warning(f"  {ticker}: parquet not found at {co_path} — skipping")
            continue

        co_df = load_parquet(co_path)
        eq_feats = compute_equity_features(co_df, ticker_spine, ticker)

        # Targets
        price    = co_df.set_index("date")["close_adj"].reindex(ticker_spine)
        tgt_df   = compute_targets(price, ticker_spine)

        # Short interest
        si_feats = compute_short_interest_features(si_df, ticker, ticker_spine)

        # Sector relative performance
        sector_etf = GICS_MAP.get(ticker, "XLF")
        if sector_etf in sector_rets:
            ticker_ret = eq_feats["eq_log_ret_1d"]
            ticker_ret.index.name = "date"  # ensure index name matches etf_ret for subtraction
            # sector_rets[etf] is on full spine with ffilled prices → log_return.
            # reindex to ticker_spine is safe (ticker_spine ⊆ spine, exact dates).
            # If still NaN after ffill+reindex, fill with 0 (no sector movement).
            etf_ret = sector_rets[sector_etf].reindex(ticker_spine).fillna(0.0)
            sr_feats = compute_sector_relative(ticker_ret, {sector_etf: etf_ret})
        else:
            sr_feats = pd.DataFrame(index=ticker_spine)

        # Slice shared block to ticker spine
        shared_slice = shared.reindex(ticker_spine)

        # Assemble ticker panel
        ticker_panel = pd.concat([
            shared_slice,
            eq_feats,
            si_feats,
            sr_feats,
            tgt_df,
        ], axis=1)

        ticker_panel.index.name = "date"
        ticker_panel = ticker_panel.reset_index()
        ticker_panel.insert(1, "ticker", ticker)

        # Drop rows before list_date (should be clean already but belt-and-braces)
        ticker_panel = ticker_panel[ticker_panel["date"] >= list_date]

        n_rows    = len(ticker_panel)
        n_missing = ticker_panel.drop(columns=["date","ticker","y_ret_21d","y_dir_21d"]).isna().mean().mean()

        build_log.append({
            "ticker":      ticker,
            "list_date":   str(list_date.date()),
            "rows":        n_rows,
            "missing_pct": round(float(n_missing) * 100, 1),
            "sector_etf":  sector_etf,
        })

        all_panels.append(ticker_panel)
        log.info(f"  ✓ {ticker:<12} rows={n_rows:5d} | missing={n_missing:.1%}")

    # ── Stack and save ─────────────────────────────────────────────────────
    log.info("Stacking all ticker panels ...")
    panel = pd.concat(all_panels, axis=0, ignore_index=True)
    panel = panel.sort_values(["date", "ticker"]).reset_index(drop=True)

    # Enforce consistent dtypes
    panel["date"]      = pd.to_datetime(panel["date"])
    panel["ticker"]    = panel["ticker"].astype("category")
    panel["y_dir_21d"] = panel["y_dir_21d"].astype("float32")

    out_path = FEATURES_DIR / "panel_raw.parquet"
    panel.to_parquet(out_path, index=False, engine="pyarrow", compression="snappy")

    feature_cols = [c for c in panel.columns
                    if c not in ("date","ticker","y_ret_21d","y_dir_21d")]

    summary = {
        "generated_at":    datetime.utcnow().isoformat(),
        "spine_start":     str(spine[0].date()),
        "spine_end":       str(spine[-1].date()),
        "trading_days":    len(spine),
        "tickers":         len(all_panels),
        "total_rows":      len(panel),
        "feature_cols":    len(feature_cols),
        "output":          str(out_path),
        "per_ticker":      build_log,
    }

    log_path = FEATURES_DIR / "panel_build_log.json"
    with open(log_path, "w") as f:
        json.dump(summary, f, indent=2)

    log.info("=" * 65)
    log.info("PANEL BUILD COMPLETE")
    log.info(f"  Rows          : {len(panel):,}")
    log.info(f"  Tickers       : {len(all_panels)}")
    log.info(f"  Feature cols  : {len(feature_cols)}")
    log.info(f"  Output        : {out_path}")
    log.info(f"  Build log     : {log_path}")
    log.info("=" * 65)


def verify() -> None:
    out_path = FEATURES_DIR / "panel_raw.parquet"
    if not out_path.exists():
        log.error(f"panel_raw.parquet not found at {out_path}")
        sys.exit(1)

    panel = pd.read_parquet(out_path)
    panel["date"] = pd.to_datetime(panel["date"])

    feature_cols = [c for c in panel.columns
                    if c not in ("date","ticker","y_ret_21d","y_dir_21d")]

    log.info("=" * 65)
    log.info("VERIFICATION — panel_raw.parquet")
    log.info(f"  Rows         : {len(panel):,}")
    log.info(f"  Tickers      : {panel['ticker'].nunique()}")
    log.info(f"  Feature cols : {len(feature_cols)}")
    log.info(f"  Date range   : {panel['date'].min().date()} → {panel['date'].max().date()}")
    log.info(f"  Target (y_ret_21d) non-null: {panel['y_ret_21d'].notna().sum():,}")
    log.info(f"  Target (y_dir_21d) non-null: {panel['y_dir_21d'].notna().sum():,}")

    # Check no rows before list_date per ticker
    universe = load_universe()
    violations = 0
    for ticker, info in universe["tickers"].items():
        list_date = pd.Timestamp(info.get("list_date", "2005-01-01"))
        sub = panel[panel["ticker"] == ticker]
        pre = sub[sub["date"] < list_date]
        if len(pre) > 0:
            log.warning(f"  ✗ {ticker}: {len(pre)} rows before list_date {list_date.date()}")
            violations += 1

    if violations == 0:
        log.info("  ✓ No pre-list_date rows found")

    # Missing rate summary
    missing = panel[feature_cols].isna().mean()
    high_missing = missing[missing > 0.5]
    if len(high_missing) > 0:
        log.warning(f"  Features with >50% missing ({len(high_missing)}): {list(high_missing.index[:10])}")
    else:
        log.info(f"  ✓ No features with >50% missing")

    log.info("=" * 65)


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--verify", action="store_true",
                        help="Verify existing panel_raw.parquet without rebuilding")
    parser.add_argument("--ticker", default=None, nargs="+",
                        help="Build for specific ticker(s) only, e.g. --ticker CBA.AX BHP.AX")
    args = parser.parse_args()

    if args.verify:
        verify()
    else:
        build(tickers_filter=args.ticker)