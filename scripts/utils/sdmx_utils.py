"""
sdmx_utils.py — ABS SDMX normalisation.
Fixes the time_period → date bug that broke all 5 ABS files.
"""
import pandas as pd

def normalise_sdmx(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for old, new in [("time_period","date"),("TIME_PERIOD","date"),
                     ("obs_value","value"),("OBS_VALUE","value")]:
        if old in df.columns:
            df = df.rename(columns={old: new})
    if "date" not in df.columns:
        raise ValueError(f"No date column after normalise. Got: {list(df.columns)}")
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"]).sort_values("date")
    df["date"] = df["date"].dt.normalize()
    return df
