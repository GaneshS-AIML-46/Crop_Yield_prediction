"""
Phase 1 — Data Cleaning & Preprocessing
Full pipeline: inspection -> whitespace fix -> missing values -> outliers -> duplicates -> dtypes
"""
import pandas as pd
import numpy as np
from pathlib import Path
from scipy.stats.mstats import winsorize

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def load_data():
    return pd.read_csv(PROJECT_ROOT / "crop_yield.csv")


def strip_whitespace(df):
    """Step 1.2 - Strip leading/trailing whitespace from string columns."""
    for col in df.select_dtypes(include=["object"]).columns:
        df[col] = df[col].astype(str).str.strip()
    return df


def handle_missing_values(df):
    """Step 1.3 - Fill nulls with group medians, drop rows where Yield is null."""
    # Drop rows where Yield is null
    before_nulls = df.isnull().sum()
    df = df.dropna(subset=["Yield"])
    
    numeric_cols = ["Area", "Production", "Annual_Rainfall", "Fertilizer", "Pesticide", "Yield"]
    
    for col in ["Area", "Production"]:
        if col in df.columns and df[col].isnull().any():
            df[col] = df.groupby(["Crop", "Season"])[col].transform(
                lambda x: x.fillna(x.median())
            )
    
    for col in ["Annual_Rainfall", "Fertilizer", "Pesticide"]:
        if col in df.columns and df[col].isnull().any():
            df[col] = df.groupby(["State", "Crop_Year"])[col].transform(
                lambda x: x.fillna(x.median())
            )
    
    return df, before_nulls


def cap_outliers_per_crop(df, target_col="Yield"):
    """Step 1.4 - Winsorize Yield at 1st and 99th percentile per Crop group."""
    def cap_group(series):
        if len(series) < 10:
            return series
        low, high = np.percentile(series, [1, 99])
        return series.clip(low, high)
    
    df = df.copy()
    df[target_col] = df.groupby("Crop", group_keys=False)[target_col].transform(cap_group)
    return df


def remove_duplicates(df):
    """Step 1.5 - Remove duplicates on Crop, Crop_Year, Season, State."""
    before = len(df)
    df = df.drop_duplicates(subset=["Crop", "Crop_Year", "Season", "State"], keep="first")
    return df, before - len(df)


def fix_dtypes(df):
    """Step 1.6 - Ensure correct dtypes."""
    df["Crop_Year"] = df["Crop_Year"].astype("int64")
    for col in ["Area", "Production", "Annual_Rainfall", "Fertilizer", "Pesticide", "Yield"]:
        df[col] = pd.to_numeric(df[col], errors="coerce").astype("float64")
    for col in ["Crop", "Season", "State"]:
        df[col] = df[col].astype("string")
    return df


def run_cleaning_pipeline():
    df = load_data()
    print("Phase 1: Data Cleaning")
    print("=" * 50)
    
    # 1.2 Strip whitespace
    df = strip_whitespace(df)
    print("1.2 Stripped whitespace from string columns")
    print(f"    Season unique: {sorted(df['Season'].unique())}")
    
    # 1.3 Missing values
    df, before_nulls = handle_missing_values(df)
    print("\n1.3 Missing values handled (Yield nulls dropped if any)")
    
    # 1.4 Outlier capping (per crop)
    df = cap_outliers_per_crop(df)
    print("\n1.4 Yield capped at 1st-99th percentile per Crop")
    
    # 1.5 Duplicates
    df, dup_removed = remove_duplicates(df)
    print(f"\n1.5 Duplicates removed: {dup_removed}")
    
    # 1.6 Dtypes
    df = fix_dtypes(df)
    print("\n1.6 Dtypes fixed")
    print(df.dtypes)
    
    return df


if __name__ == "__main__":
    (PROJECT_ROOT / "data").mkdir(exist_ok=True)
    df_clean = run_cleaning_pipeline()
    df_clean.to_csv(PROJECT_ROOT / "data" / "crop_yield_cleaned.csv", index=False)
    print(f"\nSaved cleaned data: {len(df_clean)} rows")
