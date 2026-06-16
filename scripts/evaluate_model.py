"""
============================================================
  Crop Yield Model -- Comprehensive Evaluation Report
  Checks: Accuracy, Overfitting, Cross-validation,
          Feature Importance, Per-Tier Residual Analysis
============================================================
"""
import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import json
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# ── Paths ────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))

# ── Helpers ──────────────────────────────────────────────────────────────────
def header(title):
    print("\n" + "=" * 60)
    print(f"  {title}")
    print("=" * 60)

def metrics_dict(y_true, y_pred):
    r2   = r2_score(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae  = mean_absolute_error(y_true, y_pred)
    pct20 = np.mean(np.abs(y_pred - y_true) / (np.abs(y_true) + 1e-9) < 0.20) * 100
    pct10 = np.mean(np.abs(y_pred - y_true) / (np.abs(y_true) + 1e-9) < 0.10) * 100
    pct50 = np.mean(np.abs(y_pred - y_true) / (np.abs(y_true) + 1e-9) < 0.50) * 100
    return dict(R2=r2, RMSE=rmse, MAE=mae,
                within_10pct=pct10, within_20pct=pct20, within_50pct=pct50)

def fmt(d):
    r2_label = '✅ Good' if d['R2'] >= 0.80 else '⚠️  Below target (< 0.80)'
    return (f"  R²             = {d['R2']:.4f}   {r2_label}\n"
            f"  RMSE           = {d['RMSE']:.4f}\n"
            f"  MAE            = {d['MAE']:.4f}\n"
            f"  Within ±10%    = {d['within_10pct']:.1f}%\n"
            f"  Within ±20%    = {d['within_20pct']:.1f}%\n"
            f"  Within ±50%    = {d['within_50pct']:.1f}%")

# ── 1. Load artifacts ─────────────────────────────────────────────────────────
header("1. Loading Data & Artifacts")

from feature_engineering import run_feature_engineering

df = pd.read_csv(PROJECT_ROOT / "data" / "crop_yield_cleaned.csv")

crop_enc  = joblib.load(PROJECT_ROOT / "models" / "crop_encoder.pkl")
state_enc = joblib.load(PROJECT_ROOT / "models" / "state_encoder.pkl")
scaler    = joblib.load(PROJECT_ROOT / "models" / "scaler.pkl")
model     = joblib.load(PROJECT_ROOT / "models" / "crop_yield_model.pkl")

with open(PROJECT_ROOT / "models" / "feature_names.json") as f:
    feature_cols = json.load(f)

# Check if log-transform was used
config_path = PROJECT_ROOT / "models" / "model_config.json"
LOG_TRANSFORM = False
if config_path.exists():
    with open(config_path) as f:
        cfg = json.load(f)
    LOG_TRANSFORM = cfg.get("log_transform", False)

print(f"  Log-transform mode : {'✅ YES (log1p/expm1)' if LOG_TRANSFORM else 'NO (raw yield)'}")

df, _, _, _ = run_feature_engineering(df, fit_encoders=False,
                                       crop_enc=crop_enc, state_enc=state_enc)
for col in feature_cols:
    if col not in df.columns:
        df[col] = 0

X = df[feature_cols]
y = df["Yield"]  # original scale always

# Recreate same splits as training
y_target = np.log1p(y) if LOG_TRANSFORM else y

X_temp, X_test, yt_temp, yt_test = train_test_split(X, y_target, test_size=0.2, random_state=42, shuffle=True)
X_train, X_val, yt_train, yt_val = train_test_split(X_temp, yt_temp, test_size=0.2, random_state=42, shuffle=True)

y_test_orig  = y.loc[X_test.index]
y_train_orig = y.loc[X_train.index]
y_val_orig   = y.loc[X_val.index]

X_train_s = scaler.transform(X_train)
X_val_s   = scaler.transform(X_val)
X_test_s  = scaler.transform(X_test)

print(f"  Train  : {len(X_train):>6} rows")
print(f"  Val    : {len(X_val):>6} rows")
print(f"  Test   : {len(X_test):>6} rows")
print(f"  Features: {len(feature_cols)}")
print(f"  Yield range: {y.min():.2f} -> {y.max():.2f}  (mean={y.mean():.2f})")


def predict_original(X_scaled):
    """Predict and inverse-transform to original yield units."""
    p = model.predict(X_scaled).astype(float)
    if LOG_TRANSFORM:
        p = np.expm1(p)
    return np.clip(p, 0, None)


# ── 2. Core Accuracy Metrics (original space) ─────────────────────────────────
header("2. Model Accuracy — Original Yield Units (Train / Val / Test)")

train_m = metrics_dict(y_train_orig.values, predict_original(X_train_s))
val_m   = metrics_dict(y_val_orig.values,   predict_original(X_val_s))
test_m  = metrics_dict(y_test_orig.values,  predict_original(X_test_s))

print("\n📊 TRAIN SET")
print(fmt(train_m))
print("\n📊 VALIDATION SET")
print(fmt(val_m))
print("\n📊 TEST SET  (unseen data)")
print(fmt(test_m))

# ── 3. Overfitting Diagnosis ──────────────────────────────────────────────────
header("3. Overfitting Diagnosis")

gap    = train_m["R2"] - test_m["R2"]
rmse_g = test_m["RMSE"] - train_m["RMSE"]

print(f"\n  Train R²    = {train_m['R2']:.4f}")
print(f"  Test  R²    = {test_m['R2']:.4f}")
print(f"  Gap (Train - Test R²) = {gap:.4f}")

if gap < 0.05:
    verdict = "✅  NO significant overfitting  (gap < 0.05)"
elif gap < 0.10:
    verdict = "⚠️   MILD overfitting  (gap 0.05-0.10)"
else:
    verdict = "❌  STRONG overfitting  (gap > 0.10)"

print(f"\n  Verdict: {verdict}")
print(f"  RMSE train: {train_m['RMSE']:.4f}  |  RMSE test: {test_m['RMSE']:.4f}  |  increase: {rmse_g:.4f}")

# ── 4. Cross-Validation ───────────────────────────────────────────────────────
header("4. 5-Fold Cross-Validation (log space if applicable)")

X_all_s = scaler.transform(X)
y_all_target = np.log1p(y).values if LOG_TRANSFORM else y.values

kf = KFold(n_splits=5, shuffle=True, random_state=42)
cv_r2   = cross_val_score(model, X_all_s, y_all_target, cv=kf, scoring="r2", n_jobs=1)
cv_neg  = cross_val_score(model, X_all_s, y_all_target, cv=kf,
                          scoring="neg_root_mean_squared_error", n_jobs=1)
cv_rmse = -cv_neg

print(f"\n  Fold R²    : {[f'{v:.4f}' for v in cv_r2]}")
print(f"  Mean R²    : {cv_r2.mean():.4f}  +/- {cv_r2.std():.4f}")
print(f"  Fold RMSE  : {[f'{v:.4f}' for v in cv_rmse]}")
print(f"  Mean RMSE  : {cv_rmse.mean():.4f}  +/- {cv_rmse.std():.4f}")

if cv_r2.std() < 0.05:
    print("\n  ✅  CV scores are stable — model generalises consistently.")
else:
    print("\n  ⚠️   CV score variance is high — model may be sensitive to data splits.")

# ── 5. Feature Importance ─────────────────────────────────────────────────────
header("5. Top-10 Feature Importances")

if hasattr(model, "feature_importances_"):
    imp   = pd.Series(model.feature_importances_, index=feature_cols).sort_values(ascending=False)
    top10 = imp.head(10)
    print()
    for feat, val in top10.items():
        bar = "█" * max(1, int(val * 200))
        print(f"  {feat:<35} {val:.4f}  {bar}")
else:
    print("  (Model does not expose feature_importances_)")

# ── 6. Per-Tier Residual Analysis ────────────────────────────────────────────
header("6. Per-Tier Residual Analysis (Test Set)")

y_pred_test = predict_original(X_test_s)
y_true_test = y_test_orig.values
residuals   = y_true_test - y_pred_test

# Define tiers based on actual yield
tiers = [
    ("Low    (0–100 kg/ha)",   (y_true_test >= 0)    & (y_true_test < 100)),
    ("Medium (100–1000 kg/ha)",(y_true_test >= 100)  & (y_true_test < 1000)),
    ("High   (1000+ kg/ha)",   (y_true_test >= 1000)),
]

print(f"\n  {'Tier':<28} {'N':>6}  {'R²':>7}  {'RMSE':>10}  {'within20%':>10}")
print(f"  {'-'*70}")
for label, mask in tiers:
    if mask.sum() == 0:
        print(f"  {label:<28}  no samples")
        continue
    yt = y_true_test[mask]
    yp = y_pred_test[mask]
    r2_t  = r2_score(yt, yp) if len(yt) > 1 else float("nan")
    rmse_t= np.sqrt(mean_squared_error(yt, yp))
    w20   = np.mean(np.abs(yp - yt) / (np.abs(yt) + 1e-9) < 0.20) * 100
    print(f"  {label:<28}  {mask.sum():>5}  {r2_t:>7.4f}  {rmse_t:>10.2f}  {w20:>9.1f}%")

# Max error sample
worst_idx = np.argmax(np.abs(residuals))
print(f"\n  Worst prediction: actual={y_true_test[worst_idx]:.2f}, "
      f"predicted={y_pred_test[worst_idx]:.2f}, "
      f"error={residuals[worst_idx]:.2f}")

# ── 7. Overall Residual Stats ─────────────────────────────────────────────────
header("7. Overall Residual Stats (Test Set)")
print(f"\n  Mean residual : {residuals.mean():.4f}  (close to 0 is unbiased)")
print(f"  Std  residual : {residuals.std():.4f}")
print(f"  Max |error|   : {np.abs(residuals).max():.4f}")
print(f"\n  Within +/-10% : {test_m['within_10pct']:.1f}%")
print(f"  Within +/-20% : {test_m['within_20pct']:.1f}%")
print(f"  Within +/-50% : {test_m['within_50pct']:.1f}%")

# ── 8. Final Verdict ──────────────────────────────────────────────────────────
header("8. FINAL VERDICT")

print(f"""
  Model         : {cfg.get('best_model', 'RandomForest') if config_path.exists() else 'Unknown'}
  Log Transform : {'Yes (log1p -> expm1)' if LOG_TRANSFORM else 'No'}

  Test R²       : {test_m['R2']:.4f}   {'✅ Target met (>= 0.80)' if test_m['R2'] >= 0.80 else '❌ Below target'}
  CV R²         : {cv_r2.mean():.4f}   {'✅ Stable' if cv_r2.std() < 0.05 else '⚠️  Unstable'}
  Overfit Gap   : {gap:.4f}   {'✅ None' if gap < 0.05 else '⚠️  Mild' if gap < 0.10 else '❌ Strong'}
  Within +/-20% : {test_m['within_20pct']:.1f}%
""")
