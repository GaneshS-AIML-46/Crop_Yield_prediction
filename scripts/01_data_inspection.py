"""
Phase 1.1 — Initial Inspection of crop_yield.csv
"""
import pandas as pd
from pathlib import Path

# Load data (path relative to project root)
PROJECT_ROOT = Path(__file__).resolve().parent.parent
df = pd.read_csv(PROJECT_ROOT / "crop_yield.csv")

print("=" * 60)
print("CROP YIELD DATASET — INITIAL INSPECTION")
print("=" * 60)
print("\n1. SHAPE")
print(f"   Rows: {len(df)}, Columns: {len(df.columns)}")

print("\n2. DATA TYPES (dtypes)")
print(df.dtypes)

print("\n3. NULL COUNTS PER COLUMN")
null_counts = df.isnull().sum()
print(null_counts[null_counts > 0] if null_counts.sum() > 0 else "   No nulls found")
print(null_counts)

print("\n4. DUPLICATE ROWS (Crop + Crop_Year + Season + State)")
dup_cols = ["Crop", "Crop_Year", "Season", "State"]
dup_count = df.duplicated(subset=dup_cols, keep=False).sum()
print(f"   Duplicate rows: {dup_count}")

print("\n5. BASIC describe() STATS")
print(df.describe())

print("\n6. UNIQUE VALUE COUNTS — CATEGORICAL COLUMNS")
for col in ["Crop", "Season", "State"]:
    print(f"\n   {col}: {df[col].nunique()} unique values")
    # Show sample values with repr to reveal whitespace
    samples = df[col].unique()[:15].tolist()
    print(f"   Sample (repr): {[repr(s) for s in samples]}")

print("\n7. YIELD OUTLIERS — Top 10 highest")
print(df.nlargest(10, "Yield")[["Crop", "State", "Yield", "Area", "Production"]])

print("\n8. ANOMALIES FLAGGED")
if null_counts.sum() > 0:
    print(f"   [!] Null values detected in: {null_counts[null_counts > 0].index.tolist()}")
if dup_count > 0:
    print(f"   [!] {dup_count} duplicate rows (Crop+Year+Season+State)")
max_yield = df["Yield"].max()
if max_yield > 500:
    print(f"   [!] Extreme Yield outlier: {max_yield} (Coconut - different units)")
season_with_spaces = df["Season"].apply(lambda x: x != str(x).strip() if pd.notna(x) else False)
if season_with_spaces.any():
    print(f"   [!] Trailing/leading whitespace in Season: {season_with_spaces.sum()} rows")
