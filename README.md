# Financial-data

Commercial-grade financial data pipeline for ASX equity forecasting.
Feeds PatchTST, DLinear, N-BEATS, and TFT models.

## Structure
```
config/          — All configuration (data, features, model, universe)
scripts/
  fetch/         — One script per data source
  clean/         — Validation and cleaning
  compute/       — Feature engineering
  build/         — Master dataset assembly
  validate/      — Audit and schema enforcement
  utils/         — Shared utilities (never duplicate these)
data/
  raw/           — Fetched data (git-ignored)
  processed/     — Cleaned data (git-ignored)
  features/      — Engineered features (git-ignored)
  master/        — Final model-ready datasets (git-ignored)
  registry/      — Schema locks and manifests (committed)
  splits/        — Split definitions (committed)
models/          — Model configs and results (weights git-ignored)
notebooks/       — Exploration and validation
tests/           — One test file per script
```

## Setup
```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # add your FRED API key
```

## Pipeline order
```
1. scripts/fetch/       — fetch all raw data
2. scripts/clean/       — validate and clean
3. scripts/compute/     — engineer features
4. scripts/build/       — assemble master dataset
5. scripts/validate/    — audit and lock schema
```