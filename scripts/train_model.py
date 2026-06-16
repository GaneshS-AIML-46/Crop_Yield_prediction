"""
Phases 3 & 4 — Train/Val/Test Split + ML Model Training
FIX: log1p target transform to reduce residuals on high-yield outlier crops.
"""
import pandas as pd
import numpy as np
import json
import joblib
from pathlib import Path
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Force LightGBM for deployment — smaller model file (<100MB for GitHub)
DEPLOY_MODE = True

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Try LightGBM and XGBoost
try:
    import lightgbm as lgb
    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False
try:
    import xgboost as xgb
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False


def load_cleaned_data():
    path = PROJECT_ROOT / "data" / "crop_yield_cleaned.csv"
    if not path.exists():
        raise FileNotFoundError("Run data_cleaning.py first")
    return pd.read_csv(path)


def scale_features(X_train, X_val, X_test, fit_scaler=True, scaler=None):
    if fit_scaler:
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled   = scaler.transform(X_val)
        X_test_scaled  = scaler.transform(X_test)
    else:
        X_train_scaled = scaler.transform(X_train)
        X_val_scaled   = scaler.transform(X_val)
        X_test_scaled  = scaler.transform(X_test)
    return X_train_scaled, X_val_scaled, X_test_scaled, scaler


def evaluate_log(y_true_log, y_pred_log, label=""):
    """Metrics in LOG space (used during model selection)."""
    return {
        "R2":   r2_score(y_true_log, y_pred_log),
        "RMSE": np.sqrt(mean_squared_error(y_true_log, y_pred_log)),
        "MAE":  mean_absolute_error(y_true_log, y_pred_log),
    }


def evaluate_original(y_true, y_pred_log, label=""):
    """Metrics in ORIGINAL space (used for final reporting)."""
    y_pred = np.expm1(y_pred_log)
    y_pred = np.clip(y_pred, 0, None)
    return {
        "R2":   r2_score(y_true, y_pred),
        "RMSE": np.sqrt(mean_squared_error(y_true, y_pred)),
        "MAE":  mean_absolute_error(y_true, y_pred),
        "within_20pct": float(np.mean(
            np.abs(y_pred - y_true) / (np.abs(y_true) + 1e-9) < 0.20
        ) * 100),
    }


def main():
    # ── Load data ────────────────────────────────────────────────────────────
    df = load_cleaned_data()

    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from feature_engineering import run_feature_engineering

    df, feature_cols, crop_enc, state_enc = run_feature_engineering(df, fit_encoders=True)

    X = df[feature_cols]
    y = df["Yield"]   # original scale

    # ── LOG1P TRANSFORM on target ─────────────────────────────────────────────
    y_log = np.log1p(y)
    print("Target transform: log1p applied")
    print(f"  Yield  range: {y.min():.2f} -> {y.max():.2f}  (mean={y.mean():.2f})")
    print(f"  log(Y) range: {y_log.min():.3f} -> {y_log.max():.3f}  (mean={y_log.mean():.3f})")

    # ── Phase 3: Split 64% / 16% / 20% ──────────────────────────────────────
    X_temp, X_test, y_temp_log, y_test_log = train_test_split(
        X, y_log, test_size=0.2, random_state=42, shuffle=True
    )
    X_train, X_val, y_train_log, y_val_log = train_test_split(
        X_temp, y_temp_log, test_size=0.2, random_state=42, shuffle=True
    )

    # Also keep original-scale test targets for final reporting
    _, y_test_orig = y.loc[X_temp.index].align(y.loc[X_test.index], join="right")
    y_test_orig = y.loc[X_test.index]

    X_train_s, X_val_s, X_test_s, scaler = scale_features(X_train, X_val, X_test)

    print(f"\nPhase 3: Data splits")
    print(f"  Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")

    # ── Phase 4.1: Baseline ───────────────────────────────────────────────────
    lr = LinearRegression()
    lr.fit(X_train_s, y_train_log)
    lr_train = evaluate_log(y_train_log, lr.predict(X_train_s))
    lr_test  = evaluate_log(y_test_log,  lr.predict(X_test_s))
    print(f"\nPhase 4.1: Baseline Linear Regression (log space)")
    print(f"  Train R2={lr_train['R2']:.4f}, Test R2={lr_test['R2']:.4f}")

    # ── Phase 4.2: Compare models ─────────────────────────────────────────────
    models = [
        ("RandomForest",    RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)),
        ("GradientBoosting", GradientBoostingRegressor(n_estimators=100, learning_rate=0.1,
                                                        loss="huber", random_state=42)),
    ]
    if HAS_XGBOOST:
        models.append(("XGBoost", xgb.XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42)))
    if HAS_LIGHTGBM:
        models.append(("LightGBM", lgb.LGBMRegressor(n_estimators=100, learning_rate=0.1, random_state=42)))

    results = []
    best_name, best_model, best_val_r2 = None, None, -np.inf

    print("\nPhase 4.2: Model comparison (log space)")
    for name, model in models:
        model.fit(X_train_s, y_train_log)
        train_r2 = r2_score(y_train_log, model.predict(X_train_s))
        val_r2   = r2_score(y_val_log,   model.predict(X_val_s))
        val_rmse = np.sqrt(mean_squared_error(y_val_log, model.predict(X_val_s)))
        results.append((name, train_r2, val_r2, val_rmse))
        print(f"  {name}: Train R2={train_r2:.4f}, Val R2={val_r2:.4f}, Val RMSE={val_rmse:.4f}")
        if val_r2 > best_val_r2:
            best_val_r2  = val_r2
            best_name, best_model = name, model

    # ── Phase 4.3: Optional tuning ────────────────────────────────────────────
    TUNE = False
    if TUNE and best_name in ("LightGBM", "XGBoost", "RandomForest"):
        if best_name == "LightGBM" and HAS_LIGHTGBM:
            param_dist = {"n_estimators": [100, 200], "max_depth": [5, 7], "learning_rate": [0.05, 0.1]}
            base = lgb.LGBMRegressor(random_state=42)
        elif best_name == "XGBoost" and HAS_XGBOOST:
            param_dist = {"n_estimators": [100, 200], "max_depth": [5, 7], "learning_rate": [0.05, 0.1]}
            base = xgb.XGBRegressor(random_state=42)
        else:
            param_dist = {"n_estimators": [100, 200], "max_depth": [8, 12]}
            base = RandomForestRegressor(random_state=42)
        search = RandomizedSearchCV(base, param_dist, n_iter=5, cv=3,
                                    scoring="neg_root_mean_squared_error",
                                    random_state=42, n_jobs=1)
        search.fit(X_train_s, y_train_log)
        best_model = search.best_estimator_
        print(f"\nPhase 4.3: Tuned {best_name}: {search.best_params_}")
    else:
        print(f"\nPhase 4.3: Using best model ({best_name}) without tuning")

    # Deploy mode: always save LightGBM (smaller pkl for GitHub <100MB limit)
    if DEPLOY_MODE and HAS_LIGHTGBM:
        print("\nDeploy mode: training LightGBM for smaller model file...")
        best_name = "LightGBM"
        best_model = lgb.LGBMRegressor(
            n_estimators=100, learning_rate=0.1, max_depth=8,
            random_state=42, verbose=-1
        )
        best_model.fit(X_train_s, y_train_log)

    # ── Final evaluation ──────────────────────────────────────────────────────
    train_r2_log = r2_score(y_train_log, best_model.predict(X_train_s))
    test_r2_log  = r2_score(y_test_log,  best_model.predict(X_test_s))
    orig_metrics = evaluate_original(y_test_orig.values, best_model.predict(X_test_s))

    print(f"\nFinal Results (best model: {best_name})")
    print(f"  Log-space  Train R2 : {train_r2_log:.4f}")
    print(f"  Log-space  Test  R2 : {test_r2_log:.4f}")
    print(f"  Log-space  Gap      : {train_r2_log - test_r2_log:.4f}")
    print(f"  Orig-space Test  R2 : {orig_metrics['R2']:.4f}")
    print(f"  Orig-space RMSE     : {orig_metrics['RMSE']:.4f}")
    print(f"  Within +/-20%%       : {orig_metrics['within_20pct']:.1f}%")

    # ── Save artifacts ────────────────────────────────────────────────────────
    models_dir = PROJECT_ROOT / "models"
    models_dir.mkdir(exist_ok=True)

    joblib.dump(best_model,  models_dir / "crop_yield_model.pkl")
    joblib.dump(scaler,      models_dir / "scaler.pkl")
    joblib.dump(crop_enc,    models_dir / "crop_encoder.pkl")
    joblib.dump(state_enc,   models_dir / "state_encoder.pkl")

    with open(models_dir / "feature_names.json", "w") as f:
        json.dump(feature_cols, f, indent=2)

    # Save transform config so API knows to apply expm1
    config = {
        "log_transform": True,
        "best_model": best_name,
        "test_r2_log": round(test_r2_log, 4),
        "test_r2_original": round(orig_metrics["R2"], 4),
        "within_20pct": round(orig_metrics["within_20pct"], 1),
    }
    with open(models_dir / "model_config.json", "w") as f:
        json.dump(config, f, indent=2)

    print("\nSaved: crop_yield_model.pkl, scaler.pkl, encoders, feature_names.json, model_config.json")


if __name__ == "__main__":
    main()
