#!/usr/bin/env python3
"""
fetch_companies_asx200.py
=========================
Fetches ASX 200 OHLCV + fundamental data from yfinance.

Upgrades vs the original ASX 50 script
---------------------------------------
  1. Universe: ~200 tickers (ASX 200) across 11 GICS sectors.
  2. Robust download loop: per-ticker retry with exponential backoff,
     throttle between tickers, one bad ticker never kills the run.
  3. Fundamental data: trailingEPS, forwardEPS, trailingPE, priceToBook,
     marketCap fetched via yf.Ticker.info and stored per-ticker as JSON.
     Used downstream to build the earnings revision proxy feature.
  4. Same point-in-time fetch, quarantine, and manifest pattern as original.

New feature groups (built in build_features_asx200.py, not here):
  - Sector-relative momentum    : stock return − sector median return
  - Earnings revision proxy     : (forwardEPS - trailingEPS) / |trailingEPS|
  - Historical vol term structure: realised vol at 10d / 21d / 63d + ratio
  - Extended technicals          : same as existing 260 features

Usage:
    python -m scripts.fetch.equity.fetch_companies_asx200

Output:
    data/raw/companies/<TICKER>.parquet
    data/raw/companies/<TICKER>_manifest.json
    data/raw/companies/<TICKER>_fundamentals.json   ← NEW
    data/raw/companies/_fetch_summary_asx200.json
"""

import json
import logging
import sys
import time
from datetime import date, datetime
from pathlib import Path

import pandas as pd
import yaml
import yfinance as yf
from dotenv import load_dotenv

# ── Project root ──────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.utils.yf_utils import quarantine, validate, write_manifest
from scripts.utils.canonical_map import safe_name

load_dotenv(PROJECT_ROOT / ".env")

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(PROJECT_ROOT / "logs" / "fetch_companies_asx200.log"),
    ],
)
log = logging.getLogger(__name__)

CONFIG_PATH   = PROJECT_ROOT / "config" / "data.yaml"
MANIFEST_PATH = PROJECT_ROOT / "config" / "universe_manifest.json"

# ── Fetch tuning ──────────────────────────────────────────────────────────────
RETRY_ATTEMPTS   = 3        # download attempts per ticker
RETRY_BACKOFF    = 5        # seconds to wait between retries
THROTTLE_BETWEEN = 0.5      # seconds between tickers (yfinance rate limit)

# ── Consistency guards ────────────────────────────────────────────────────────
# Tickers that fail these checks are tagged SKIPPED in the summary and
# excluded from the feature panel by build_features_asx200.py.
#
# MIN_TRAIN_ROWS: minimum trading days required in the TRAINING window
#   (2005-01-03 → 2019-12-31 = ~3,775 trading days for a full-history ticker).
#   We require at least 3 years (≈756 days) so rolling 252-day features
#   have enough warmup without excessive NaN padding.
MIN_TRAIN_ROWS = 756          # ~3 years of trading days in train window

# MIN_COVERAGE_PCT: fraction of expected trading days that must be present.
#   Handles delisted / trading-halt gaps. 0.90 allows ~25 missing days/year.
MIN_COVERAGE_PCT = 0.90       # 90 % of expected calendar trading days

# MAX_GAP_DAYS: largest allowed consecutive gap (calendar days).
#   > 30 days almost always indicates a suspension or major data error.
MAX_GAP_DAYS = 30             # calendar days


# ─────────────────────────────────────────────────────────────────────────────
# ASX 200 Universe  (GICS sector → ticker list)
# Add / remove tickers here as the index composition changes.
# Source: ASX 200 constituents as of March 2026.
# ─────────────────────────────────────────────────────────────────────────────

ASX200_TICKERS: dict[str, str] = {
    # ── Financials ────────────────────────────────────────────────────────────
    "CBA.AX":  "Financials",
    "NAB.AX":  "Financials",
    "WBC.AX":  "Financials",
    "ANZ.AX":  "Financials",
    "MQG.AX":  "Financials",
    "SUN.AX":  "Financials",
    "IAG.AX":  "Financials",
    "AMP.AX":  "Financials",
    "ASX.AX":  "Financials",
    "CPU.AX":  "Financials",
    "CCP.AX":  "Financials",
    "BOQ.AX":  "Financials",
    "BEN.AX":  "Financials",
    "PPT.AX":  "Financials",
    "HUB.AX":  "Financials",   # NB: listed under Technology in yfinance but ASX sector = Financials
    "IFL.AX":  "Financials",
    # PDL.AX  → NO_DATA (delisted / ticker changed)
    # PTM.AX  → NO_DATA (delisted 2024)
    "GQG.AX":  "Financials",   # SKIPPED for training (listed 2021), kept for inference
    # ── Materials ─────────────────────────────────────────────────────────────
    "BHP.AX":  "Materials",
    "RIO.AX":  "Materials",
    "FMG.AX":  "Materials",
    "S32.AX":  "Materials",
    "AMC.AX":  "Materials",
    "JHX.AX":  "Materials",
    "ORI.AX":  "Materials",
    "NEM.AX":  "Materials",    # SKIPPED for training (listed 2023), kept for inference
    "MIN.AX":  "Materials",
    "ILU.AX":  "Materials",
    # AWC.AX  → NO_DATA (delisted — Alumina Ltd acquired by Alcoa 2024)
    "IGO.AX":  "Materials",
    "PLS.AX":  "Materials",
    # LTM.AX  → NO_DATA (ticker invalid / not on yfinance)
    "SGM.AX":  "Materials",
    "BSL.AX":  "Materials",
    "CIA.AX":  "Materials",
    "GRR.AX":  "Materials",
    "LYC.AX":  "Materials",
    "NIC.AX":  "Materials",    # SKIPPED for training (listed 2018), kept for inference
    # OZL.AX  → NO_DATA (delisted — OZ Minerals acquired by BHP 2023)
    "SFR.AX":  "Materials",
    # ── Energy ────────────────────────────────────────────────────────────────
    "WDS.AX":  "Energy",
    "STO.AX":  "Energy",
    "ALD.AX":  "Energy",
    "KAR.AX":  "Energy",
    "BPT.AX":  "Energy",
    "NHC.AX":  "Energy",
    "WHC.AX":  "Energy",
    "VEA.AX":  "Energy",       # SKIPPED for training (listed 2018), kept for inference
    # ── Utilities ─────────────────────────────────────────────────────────────
    "ORG.AX":  "Utilities",
    "AGL.AX":  "Utilities",
    "APA.AX":  "Utilities",
    # SKI.AX  → NO_DATA (delisted — Spark Infrastructure acquired 2021)
    # ── Industrials ───────────────────────────────────────────────────────────
    "TCL.AX":  "Industrials",
    "QAN.AX":  "Industrials",
    "BXB.AX":  "Industrials",
    "AIA.AX":  "Industrials",
    "AZJ.AX":  "Industrials",
    # SYD.AX  → NO_DATA (delisted — Sydney Airport taken private 2022)
    "DOW.AX":  "Industrials",
    # CIM.AX  → NO_DATA (ticker invalid)
    "QUB.AX":  "Industrials",
    "SIQ.AX":  "Industrials",
    "WOR.AX":  "Industrials",
    "FLT.AX":  "Industrials",
    "MND.AX":  "Industrials",
    "SRG.AX":  "Industrials",
    "CWY.AX":  "Industrials",
    # ── Real Estate ───────────────────────────────────────────────────────────
    "GPT.AX":  "Real Estate",
    "MGR.AX":  "Real Estate",
    "SCG.AX":  "Real Estate",
    "NSR.AX":  "Real Estate",
    "LLC.AX":  "Real Estate",
    "DXS.AX":  "Real Estate",
    "GMG.AX":  "Real Estate",
    "CHC.AX":  "Real Estate",
    "CLW.AX":  "Real Estate",
    "HDN.AX":  "Real Estate",  # SKIPPED for training (listed 2020), kept for inference
    "BWP.AX":  "Real Estate",
    "CIP.AX":  "Real Estate",
    "CNI.AX":  "Real Estate",
    "GDI.AX":  "Real Estate",
    "HMC.AX":  "Real Estate",  # SKIPPED for training (listed 2021), kept for inference
    # ABP.AX  → NO_DATA (delisted — Abacus Property demerged/restructured)
    "ARF.AX":  "Real Estate",
    "CQR.AX":  "Real Estate",
    "PXA.AX":  "Real Estate",  # SKIPPED for training (listed 2021), kept for inference
    "WAM.AX":  "Real Estate",
    # ── Health Care ───────────────────────────────────────────────────────────
    "CSL.AX":  "Health Care",
    "RHC.AX":  "Health Care",
    "SHL.AX":  "Health Care",
    "COH.AX":  "Health Care",
    "PME.AX":  "Health Care",
    "RMD.AX":  "Health Care",
    "NVX.AX":  "Health Care",
    "ACL.AX":  "Health Care",  # SKIPPED for training (listed 2021), kept for inference
    # API.AX  → NO_DATA (delisted — Australian Pharmaceutical acquired 2021)
    "ARX.AX":  "Health Care",  # SKIPPED for training (listed 2020), kept for inference
    # AWF.AX  → NO_DATA (ticker invalid)
    "CUV.AX":  "Health Care",
    "EBR.AX":  "Health Care",  # SKIPPED for training (listed 2021), kept for inference
    "MSB.AX":  "Health Care",
    "PNV.AX":  "Health Care",
    "RAC.AX":  "Health Care",
    # ── Consumer Discretionary ────────────────────────────────────────────────
    "JBH.AX":  "Consumer Discretionary",
    "DMP.AX":  "Consumer Discretionary",
    "SUL.AX":  "Consumer Discretionary",
    "HVN.AX":  "Consumer Discretionary",
    "PMV.AX":  "Consumer Discretionary",
    "NCK.AX":  "Consumer Discretionary",
    # BWX.AX  → NO_DATA (delisted / administration)
    "MYR.AX":  "Consumer Discretionary",
    "APE.AX":  "Consumer Discretionary",
    "ARB.AX":  "Consumer Discretionary",
    "BRG.AX":  "Consumer Discretionary",
    # CAI.AX  → NO_DATA (ticker invalid)
    "CKF.AX":  "Consumer Discretionary",
    "DDR.AX":  "Consumer Discretionary",
    "EVT.AX":  "Consumer Discretionary",
    # GUD.AX  → NO_DATA (may have changed ticker — check manually)
    "KGN.AX":  "Consumer Discretionary",
    "LOV.AX":  "Consumer Discretionary",
    "MTO.AX":  "Consumer Discretionary",
    "RFG.AX":  "Consumer Discretionary",  # SKIPPED for training (listed 2024)
    # SWM.AX  → NO_DATA (delisted — Seven West Media privatised)
    # TRS.AX  → NO_DATA (ticker invalid / delisted)
    # ── Consumer Staples ──────────────────────────────────────────────────────
    "WES.AX":  "Consumer Staples",
    "WOW.AX":  "Consumer Staples",
    "COL.AX":  "Consumer Staples",   # SKIPPED for training (listed 2018), kept for inference
    "MTS.AX":  "Consumer Staples",
    "TWE.AX":  "Consumer Staples",
    "GNC.AX":  "Consumer Staples",
    "ING.AX":  "Consumer Staples",
    "CGF.AX":  "Consumer Staples",
    "CCL.AX":  "Consumer Staples",   # SKIPPED for training (listed 2024), kept for inference
    "GWA.AX":  "Consumer Staples",
    "NZM.AX":  "Consumer Staples",
    "UNI.AX":  "Consumer Staples",   # SKIPPED for training (listed 2020), kept for inference
    # ── Communication Services ────────────────────────────────────────────────
    "TLS.AX":  "Communication Services",
    "SEK.AX":  "Communication Services",
    "CAR.AX":  "Communication Services",
    "TPG.AX":  "Communication Services",
    "NWS.AX":  "Communication Services",  # SKIPPED — 569-day data gap
    "REA.AX":  "Communication Services",
    "IEL.AX":  "Communication Services",
    # SCA.AX  → NO_DATA (ticker invalid / changed)
    "CNU.AX":  "Communication Services",
    "MCE.AX":  "Communication Services",
    "OML.AX":  "Communication Services",
    # ── Technology ────────────────────────────────────────────────────────────
    "WTC.AX":  "Technology",
    "XRO.AX":  "Technology",
    # ALU.AX  → NO_DATA (ticker invalid on yfinance — try "ALU.AX" manually)
    "APX.AX":  "Technology",
    "MP1.AX":  "Technology",
    "NXT.AX":  "Technology",
    "BVS.AX":  "Technology",
    "DTL.AX":  "Technology",
    "EML.AX":  "Technology",
    "HUB.AX":  "Technology",
    "IRI.AX":  "Technology",
    # LNK.AX  → NO_DATA (delisted — Link Administration acquired 2023)
    "MAQ.AX":  "Technology",
    "NDQ.AX":  "Technology",
    "PWH.AX":  "Technology",
    "RPL.AX":  "Technology",   # SKIPPED for training (listed 2019), kept for inference
    "SPZ.AX":  "Technology",
    "TNE.AX":  "Technology",
    # VHT.AX  → NO_DATA (ticker invalid / delisted)
    # Z1P.AX  → NO_DATA (rebranded to ZIP.AX in 2022)
    "ZIP.AX":  "Technology",
}


# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────

def load_config() -> dict:
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


def load_universe_manifest() -> dict:
    if MANIFEST_PATH.exists():
        with open(MANIFEST_PATH) as f:
            return json.load(f)
    return {"tickers": {}}


# ─────────────────────────────────────────────────────────────────────────────
# Cleaning  (same as original, kept here to avoid import changes)
# ─────────────────────────────────────────────────────────────────────────────

def clean_yf(df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()

    df = df.copy().reset_index()

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] if c[1] in ("", ticker) else f"{c[0]}_{c[1]}"
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

    keep = ["date", "open", "high", "low", "close", "close_adj",
            "volume", "dividends", "stock_splits"]
    return df[[c for c in keep if c in df.columns]]


# ─────────────────────────────────────────────────────────────────────────────
# Fundamentals fetch  (NEW)
# ─────────────────────────────────────────────────────────────────────────────

FUNDAMENTAL_FIELDS = [
    "trailingEPS",    # last reported EPS
    "forwardEPS",     # consensus forward EPS (used for revision proxy)
    "trailingPE",     # trailing P/E
    "forwardPE",      # forward P/E (revision signal: lower → positive revision)
    "priceToBook",    # P/B ratio
    "marketCap",      # size factor proxy
    "returnOnEquity", # ROE
    "debtToEquity",   # leverage
    "revenueGrowth",  # revenue growth yoy
    "earningsGrowth", # earnings growth yoy
    "dividendYield",  # income factor
    "beta",           # market beta
]


def fetch_fundamentals(ticker: str) -> dict:
    """
    Fetch current fundamental snapshot via yf.Ticker.info.

    Returns a dict with the fields above (NaN where unavailable).
    This is NOT point-in-time — it is the current values as of fetch date.
    Used to build the earnings revision proxy feature ONLY for recent data;
    historical revision momentum must be approximated from price patterns.

    NOTE: yfinance .info can be slow and rate-limited. We catch all
    exceptions and return empty fundamentals rather than failing the run.
    """
    try:
        info = yf.Ticker(ticker).info
        data = {field: info.get(field, None) for field in FUNDAMENTAL_FIELDS}
        data["fetch_date"] = date.today().isoformat()
        data["ticker"] = ticker

        # Compute earnings revision proxy: (forwardEPS - trailingEPS) / |trailingEPS|
        fwd = data.get("forwardEPS")
        trail = data.get("trailingEPS")
        if fwd is not None and trail is not None and trail != 0:
            data["eps_revision_pct"] = (fwd - trail) / abs(trail)
        else:
            data["eps_revision_pct"] = None

        return data

    except Exception as exc:
        log.warning(f"[{ticker}] Fundamentals fetch failed: {exc}")
        return {"ticker": ticker, "fetch_date": date.today().isoformat(),
                "error": str(exc)}


# ─────────────────────────────────────────────────────────────────────────────
# OHLCV fetch  (upgraded: retries + throttle)
# ─────────────────────────────────────────────────────────────────────────────

def check_ticker_consistency(
    df: pd.DataFrame,
    ticker: str,
    global_start: str,
    train_end:    str = "2019-12-31",
) -> dict:
    """
    Run three consistency checks on the fetched OHLCV data.

    Returns a dict:
      {
        "passed":           bool   — True if all checks pass
        "train_rows":       int    — trading days in train window
        "coverage_pct":     float  — fraction of expected trading days present
        "max_gap_days":     int    — largest calendar-day gap in price series
        "skip_reason":      str    — human-readable reason if failed, else ""
      }

    Checks
    ------
    1. MIN_TRAIN_ROWS
       The ticker must have at least MIN_TRAIN_ROWS rows inside the training
       window (global_start → train_end). Tickers listed after ~2016 will
       fail this check and be excluded from training — they can still be
       used for live inference if they have sufficient recent data.

    2. MIN_COVERAGE_PCT
       Over the ticker's own date range, at least MIN_COVERAGE_PCT of the
       expected trading days must be present. We estimate expected days as
       calendar_days × (5/7) × 0.97 (adjusting for weekends and ~10 ASX
       public holidays per year).

    3. MAX_GAP_DAYS
       No consecutive gap in the price series may exceed MAX_GAP_DAYS
       calendar days. Gaps larger than this indicate a trading halt or
       a major data error that would corrupt rolling features.
    """
    result = {
        "passed":       True,
        "train_rows":   0,
        "coverage_pct": 0.0,
        "max_gap_days": 0,
        "skip_reason":  "",
    }

    if df.empty:
        result["passed"]      = False
        result["skip_reason"] = "empty DataFrame"
        return result

    dates = pd.to_datetime(df["date"]).sort_values().reset_index(drop=True)

    # ── Check 1: training-window row count ───────────────────────────────────
    train_mask = (dates >= pd.Timestamp(global_start)) & \
                 (dates <= pd.Timestamp(train_end))
    train_rows = int(train_mask.sum())
    result["train_rows"] = train_rows

    if train_rows < MIN_TRAIN_ROWS:
        result["passed"]      = False
        result["skip_reason"] = (
            f"only {train_rows} training-window rows "
            f"(need ≥ {MIN_TRAIN_ROWS}). "
            f"Ticker likely listed after {pd.Timestamp(global_start).year + 3}."
        )
        # Still compute coverage so the summary is informative
        if len(dates) > 1:
            date_min = dates.iloc[0]
            date_max = dates.iloc[-1]
            calendar_span = (date_max - date_min).days
            expected_days = calendar_span * (5 / 7) * 0.97
            result["coverage_pct"] = round(len(dates) / max(expected_days, 1), 4)
            gaps = dates.diff().dt.days.dropna()
            result["max_gap_days"] = int(gaps.max()) if len(gaps) > 0 else 0
        return result   # no point checking further

    # ── Check 2: coverage % ──────────────────────────────────────────────────
    date_min = dates.iloc[0]
    date_max = dates.iloc[-1]
    calendar_span = (date_max - date_min).days
    # Expected trading days: weekdays only × holiday adjustment
    expected_days = calendar_span * (5 / 7) * 0.97
    actual_days   = len(dates)
    coverage = actual_days / max(expected_days, 1)
    result["coverage_pct"] = round(coverage, 4)

    if coverage < MIN_COVERAGE_PCT:
        result["passed"]      = False
        result["skip_reason"] = (
            f"coverage {coverage:.1%} < {MIN_COVERAGE_PCT:.0%}. "
            f"Expected ≈{int(expected_days)} days, got {actual_days}. "
            "Likely gaps from suspensions or data issues."
        )
        return result

    # ── Check 3: max consecutive gap ─────────────────────────────────────────
    gaps = dates.diff().dt.days.dropna()
    max_gap = int(gaps.max()) if len(gaps) > 0 else 0
    result["max_gap_days"] = max_gap

    if max_gap > MAX_GAP_DAYS:
        result["passed"]      = False
        result["skip_reason"] = (
            f"max consecutive gap = {max_gap} calendar days "
            f"(limit {MAX_GAP_DAYS}). "
            "Indicates trading halt or data error."
        )
        return result

    return result


def fetch_one(ticker: str, start: str, end: str, out_dir: Path) -> dict:
    """
    Fetch OHLCV for one ticker with retry and backoff.

    Returns a result dict compatible with the original _fetch_summary.json
    format (so downstream scripts don't need changes).
    """
    log.info(f"Fetching {ticker} ...")

    raw = None
    last_exc = None

    for attempt in range(1, RETRY_ATTEMPTS + 1):
        try:
            raw = yf.download(
                ticker,
                start=start,
                end=end,
                auto_adjust=False,
                actions=True,
                progress=False,
            )
            if raw is not None and not raw.empty:
                break
            log.warning(f"[{ticker}] Attempt {attempt}: no data returned")
        except Exception as exc:
            last_exc = exc
            log.warning(f"[{ticker}] Attempt {attempt} error: {exc}")

        if attempt < RETRY_ATTEMPTS:
            time.sleep(RETRY_BACKOFF * attempt)

    if raw is None or raw.empty:
        log.warning(f"[{ticker}] No data after {RETRY_ATTEMPTS} attempts")
        return {
            "ticker": ticker, "status": "NO_DATA",
            "issues": [{"code": "NO_DATA", "detail": str(last_exc)}],
            "rows": 0, "date_min": None, "date_max": None,
        }

    df = clean_yf(raw, ticker)

    if df.empty:
        return {
            "ticker": ticker, "status": "EMPTY_AFTER_CLEAN",
            "issues": [{"code": "EMPTY_AFTER_CLEAN"}],
            "rows": 0, "date_min": None, "date_max": None,
        }

    issues = validate(df, ticker, instrument_type="equity")
    df     = quarantine(df)

    hard_fails = [i for i in issues if i["code"] == "NON_POSITIVE_PRICE"]
    status = "FAIL" if hard_fails else "WARN" if issues else "OK"

    dest = out_dir / "quarantine" if status == "FAIL" else out_dir
    dest.mkdir(parents=True, exist_ok=True)

    out_path = dest / f"{safe_name(ticker)}.parquet"
    df.to_parquet(out_path, index=False)

    write_manifest(str(out_path), ticker, df, issues, status)

    # ── Consistency check ─────────────────────────────────────────────────────
    consistency = check_ticker_consistency(df, ticker, start)
    if not consistency["passed"]:
        # Tag as SKIPPED — parquet is still saved for reference,
        # but build_features_asx200.py will exclude this ticker.
        status = "SKIPPED"
        log.warning(
            f"[{ticker}] SKIPPED — {consistency['skip_reason']} | "
            f"train_rows={consistency['train_rows']} "
            f"coverage={consistency['coverage_pct']:.1%} "
            f"max_gap={consistency['max_gap_days']}d"
        )
    else:
        log.info(
            f"[{ticker}] {status} | rows={len(df)} | "
            f"{df['date'].min().date()} → {df['date'].max().date()} | "
            f"issues={len(issues)} | "
            f"train_rows={consistency['train_rows']} "
            f"coverage={consistency['coverage_pct']:.1%} "
            f"max_gap={consistency['max_gap_days']}d"
        )

    return {
        "ticker":       ticker,
        "status":       status,
        "issues":       issues,
        "rows":         len(df),
        "date_min":     str(df["date"].min().date()),
        "date_max":     str(df["date"].max().date()),
        "output":       str(out_path),
        "consistency":  consistency,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    cfg      = load_config()
    universe = load_universe_manifest()

    global_start: str = cfg["data"]["start_date"]
    global_end:   str = cfg["data"]["end_date"] or date.today().isoformat()
    out_dir = PROJECT_ROOT / cfg["raw"]["companies"]
    out_dir.mkdir(parents=True, exist_ok=True)

    # Override with ASX 200 universe (ignores data.yaml companies.tickers)
    tickers = ASX200_TICKERS

    log.info("=" * 65)
    log.info("fetch_companies_asx200.py — ASX 200 OHLCV + Fundamentals fetch")
    log.info(f"Universe   : {len(tickers)} tickers")
    log.info(f"Date range : {global_start} → {global_end}")
    log.info(f"Output     : {out_dir}")
    log.info(f"yfinance   : {yf.__version__}")
    log.info("=" * 65)

    results      = []
    fundamentals = []

    for ticker, sector in tickers.items():

        # ── OHLCV ─────────────────────────────────────────────────────────────
        meta = universe.get("tickers", {}).get(ticker, {})
        list_date = meta.get("list_date", global_start)
        effective_start = max(
            pd.Timestamp(global_start),
            pd.Timestamp(list_date),
        ).strftime("%Y-%m-%d")

        if effective_start != global_start:
            log.info(f"[{ticker}] Partial history — fetching from {effective_start}")

        result = fetch_one(ticker, effective_start, global_end, out_dir)
        result["sector"]  = sector
        result["history"] = meta.get("history", "unknown")
        results.append(result)

        # ── Fundamentals ──────────────────────────────────────────────────────
        fund = fetch_fundamentals(ticker)
        fund["sector"] = sector
        fundamentals.append(fund)

        # Save fundamentals per ticker
        fund_path = out_dir / f"{safe_name(ticker)}_fundamentals.json"
        with open(fund_path, "w") as f:
            json.dump(fund, f, indent=2, default=str)

        # Throttle between tickers
        time.sleep(THROTTLE_BETWEEN)

    # ── Save consolidated fundamentals ────────────────────────────────────────
    fund_all_path = out_dir / "_fundamentals_asx200.json"
    with open(fund_all_path, "w") as f:
        json.dump(fundamentals, f, indent=2, default=str)
    log.info(f"Fundamentals saved: {fund_all_path}")

    # ── Summary ───────────────────────────────────────────────────────────────
    n_ok      = sum(1 for r in results if r["status"] == "OK")
    n_warn    = sum(1 for r in results if r["status"] == "WARN")
    n_fail    = sum(1 for r in results if r["status"] == "FAIL")
    n_skipped = sum(1 for r in results if r["status"] == "SKIPPED")
    n_err     = sum(1 for r in results if r["status"] in
                   {"NO_DATA", "DOWNLOAD_ERROR", "EMPTY_AFTER_CLEAN"})

    skipped_tickers = [
        {"ticker": r["ticker"],
         "reason": r.get("consistency", {}).get("skip_reason", ""),
         "train_rows": r.get("consistency", {}).get("train_rows", 0)}
        for r in results if r["status"] == "SKIPPED"
    ]

    summary = {
        "generated_at":     datetime.utcnow().isoformat(),
        "yfinance_version": yf.__version__,
        "global_start":     global_start,
        "global_end":       global_end,
        "consistency_thresholds": {
            "min_train_rows":    MIN_TRAIN_ROWS,
            "min_coverage_pct":  MIN_COVERAGE_PCT,
            "max_gap_days":      MAX_GAP_DAYS,
        },
        "total":            len(results),
        "ok":               n_ok,
        "warn":             n_warn,
        "fail":             n_fail,
        "skipped":          n_skipped,
        "error":            n_err,
        "skipped_tickers":  skipped_tickers,
        "tickers":          results,
    }

    summary_path = out_dir / "_fetch_summary_asx200.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    log.info("=" * 65)
    log.info("FETCH COMPLETE")
    log.info(f"  OK      : {n_ok}")
    log.info(f"  WARN    : {n_warn}")
    log.info(f"  FAIL    : {n_fail}")
    log.info(f"  SKIPPED : {n_skipped}  (consistency check failed)")
    log.info(f"  ERROR   : {n_err}")
    if skipped_tickers:
        log.info("  Skipped tickers:")
        for s in skipped_tickers:
            log.info(f"    {s['ticker']:<12} train_rows={s['train_rows']:>5}  {s['reason']}")
    log.info(f"  Summary       → {summary_path}")
    log.info(f"  Fundamentals  → {fund_all_path}")
    log.info("=" * 65)

    failed  = [r["ticker"] for r in results if r["status"] == "FAIL"]
    errored = [r["ticker"] for r in results
               if r["status"] in {"NO_DATA", "DOWNLOAD_ERROR"}]

    if failed:
        log.error(f"QUARANTINED: {failed}")
        sys.exit(1)
    if errored:
        log.warning(f"NO DATA (skipped, non-fatal): {errored}")
        # Not sys.exit(1) — ASX 200 has some illiquid tickers that may fail
        # without fundamentally breaking the panel. Log and continue.

    log.info("Fetch complete. Run build_features_asx200.py next.")


if __name__ == "__main__":
    main()