"""
yf_utils.py
Shared yfinance utilities. Import from here — never duplicate.
"""
import json, logging, datetime
import numpy as np
import pandas as pd
import yfinance as yf

log = logging.getLogger(__name__)
RET_MIN, RET_MAX = -0.99, 5.00

INSTRUMENT_TYPES = {
    "OIL": "commodity_future", "GOLD": "commodity_future",
    "SILVER": "commodity_future", "COPPER": "commodity_future",
    "IRON": "commodity_future", "BDI": "index",
    "DXY": "fx_index", "AUDUSD": "fx", "AUDJPY": "fx", "AUDCNY": "fx",
    "VIX": "volatility_index", "VVIX": "volatility_index", "MOVE": "volatility_index",
    "AXJO": "index", "GSPC": "index", "FTSE": "index",
    "N225": "index", "HSI": "index", "SSE": "index", "CSI300": "index",
    "DBC": "etf", "GDX": "etf",
}

MIN_ROWS = {
    "equity": 4500, "index": 4000, "fx": 4500, "fx_index": 4500,
    "commodity_future": 4000, "etf": 1500, "volatility_index": 3000,
}


def clean_yf(df: pd.DataFrame, symbol: str = "") -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    df = df.copy().reset_index()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] if c[1] in ("", symbol) else f"{c[0]}_{c[1]}"
                      for c in df.columns]
    df.columns = [str(c).lower().strip().replace(" ", "_") for c in df.columns]
    df = df.rename(columns={"adj_close": "close_adj"})
    date_col = next((c for c in df.columns if "date" in c), None)
    if date_col is None:
        return pd.DataFrame()
    df["date"] = pd.to_datetime(df[date_col], errors="coerce")
    df = (df.dropna(subset=["date"])
            .sort_values("date")
            .drop_duplicates("date")
            .reset_index(drop=True))
    keep = ["date","open","high","low","close","close_adj",
            "volume","dividends","stock_splits"]
    return df[[c for c in keep if c in df.columns]]


def validate(df: pd.DataFrame, symbol: str,
             instrument_type: str = "equity") -> list:
    issues = []
    price_col = "close_adj" if "close_adj" in df.columns else "close"
    if instrument_type == "equity":
        bad = int((df[price_col] <= 0).sum())
        if bad:
            issues.append({"code": "NON_POSITIVE_PRICE", "count": bad})
    rets = df[price_col].pct_change()
    extreme = (rets < RET_MIN) | (rets > RET_MAX)
    if extreme.sum():
        issues.append({"code": "EXTREME_RETURN", "count": int(extreme.sum()),
            "dates": df.loc[extreme, "date"].dt.strftime("%Y-%m-%d").tolist()})
    minimum = MIN_ROWS.get(instrument_type, 100)
    if len(df) < minimum:
        issues.append({"code": "LOW_ROW_COUNT",
            "count": len(df), "expected_min": minimum})
    return issues


def quarantine(df: pd.DataFrame) -> pd.DataFrame:
    price_col = "close_adj" if "close_adj" in df.columns else "close"
    if price_col not in df.columns:
        return df
    df = df.copy()
    rets = df[price_col].pct_change()
    mask = (rets < RET_MIN) | (rets > RET_MAX)
    df.loc[mask, price_col] = float("nan")
    return df


def write_manifest(path: str, symbol: str, df: pd.DataFrame,
                   issues: list, status: str) -> None:
    m = {"symbol": symbol, "output": path, "status": status,
         "fetched_at": datetime.datetime.utcnow().isoformat(),
         "yfinance_version": yf.__version__,
         "rows": len(df),
         "date_min": str(df["date"].min().date()) if len(df) else None,
         "date_max": str(df["date"].max().date()) if len(df) else None,
         "issues": issues}
    with open(path.replace(".parquet", "_manifest.json"), "w") as f:
        json.dump(m, f, indent=2)
