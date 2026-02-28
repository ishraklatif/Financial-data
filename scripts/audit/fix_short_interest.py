#!/usr/bin/env python3
"""
fix_short_interest.py
Quick fix: cap short_pct at 100 and inspect the 9 bad rows.
"""
import pandas as pd
from pathlib import Path

path = Path("data/raw/sentiment/short_interest/SHORT_INTEREST.parquet")
df = pd.read_parquet(path)

bad = df[df["short_pct"] > 100]
print(f"Bad rows ({len(bad)}):")
print(bad[["date", "product_code", "short_pct", "short_positions", "total_issued"]].to_string())

df["short_pct"] = df["short_pct"].clip(upper=100)
df.to_parquet(path, index=False)
print(f"\nFixed — short_pct capped at 100. Saved → {path}")