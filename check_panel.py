"""
Quick diagnostic on panel_raw.parquet:
1. Breakdown of missing % by feature group
2. List any daily features with >5% missing (unexpected)
3. Confirm target coverage
"""
import pandas as pd
import numpy as np
import sys

path = sys.argv[1]
panel = pd.read_parquet(path)
panel["date"] = pd.to_datetime(panel["date"])

feature_cols = [c for c in panel.columns if c not in ("date","ticker","y_ret_21d","y_dir_21d")]
missing = panel[feature_cols].isna().mean().sort_values(ascending=False)

# Group by prefix
prefixes = {}
for col in feature_cols:
    prefix = col.split("_")[0]
    if prefix not in prefixes:
        prefixes[prefix] = []
    prefixes[prefix].append(missing[col])

print(f"\n{'='*60}")
print(f"Panel: {len(panel):,} rows | {len(feature_cols)} feature cols")
print(f"Tickers: {list(panel['ticker'].unique())}")
print(f"{'='*60}")

print("\n--- Missing % by feature group prefix ---")
for pfx, vals in sorted(prefixes.items()):
    avg = np.mean(vals)
    max_ = np.max(vals)
    print(f"  {pfx:<20} n={len(vals):3d}  avg={avg:.1%}  max={max_:.1%}")

print("\n--- Daily features with >5% missing (unexpected) ---")
# Identify likely-daily features (exclude known low-freq prefixes)
low_freq_prefixes = {"CPI","PCEPI","FEDFUNDS","GDP","HSR","TOT","WPI","PMI","JOLTS","UMICH",
                     "UNEMP","INDPRO","F3","si"}
daily_high_missing = []
for col in feature_cols:
    pfx = col.split("_")[0]
    if pfx in low_freq_prefixes:
        continue
    if missing[col] > 0.05:
        daily_high_missing.append((col, missing[col]))

if daily_high_missing:
    for col, pct in sorted(daily_high_missing, key=lambda x: -x[1])[:30]:
        print(f"  {col:<45} {pct:.1%}")
else:
    print("  None — all daily features <5% missing ✓")

print("\n--- Target coverage ---")
print(f"  y_ret_21d non-null: {panel['y_ret_21d'].notna().sum():,} / {len(panel):,}  ({panel['y_ret_21d'].notna().mean():.1%})")
print(f"  y_dir_21d non-null: {panel['y_dir_21d'].notna().sum():,} / {len(panel):,}  ({panel['y_dir_21d'].notna().mean():.1%})")
print(f"  Expected ~{(5340-21)/5340:.1%} (last 21 rows per ticker are NaN by design)")

print("\n--- Feature count by group ---")
group_map = {}
for col in feature_cols:
    parts = col.split("_")
    # Use first two parts as group key
    grp = "_".join(parts[:2])
    group_map.setdefault(grp, 0)
    group_map[grp] += 1

for grp, cnt in sorted(group_map.items(), key=lambda x: -x[1])[:30]:
    print(f"  {grp:<35} {cnt}")

print(f"\n{'='*60}")