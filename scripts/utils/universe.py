"""
universe.py — Point-in-time universe filtering using universe_manifest.json.
Prevents future-listed tickers (e.g. NEM.AX) appearing in historical splits.
"""
import json
import pandas as pd
from pathlib import Path

MANIFEST_PATH = Path("config/universe_manifest.json")

def load_manifest(path: Path = MANIFEST_PATH) -> dict:
    with open(path) as f:
        return json.load(f)

def get_tickers_for_split(split_start: str, split_end: str,
                           path: Path = MANIFEST_PATH) -> list:
    """
    Return tickers that were listed before split_start
    and not delisted before split_end.
    """
    manifest = load_manifest(path)
    start = pd.Timestamp(split_start)
    end   = pd.Timestamp(split_end)
    valid = []
    for ticker, meta in manifest["tickers"].items():
        listed   = pd.Timestamp(meta["list_date"])
        delisted = pd.Timestamp(meta["delist_date"]) if meta["delist_date"] else pd.Timestamp.max
        if listed <= start and delisted >= end:
            valid.append(ticker)
    return sorted(valid)
