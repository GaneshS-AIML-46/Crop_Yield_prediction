"""
Phase 2 — Feature Engineering
Encoding, new features, feature selection, scaling
"""
import pandas as pd
import numpy as np
import json
from pathlib import Path
from sklearn.preprocessing import LabelEncoder, StandardScaler
import joblib

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def encode_categoricals(df, crop_encoder=None, state_encoder=None, fit=True):
    """Step 2.1 - Label encode Crop & State, One-hot encode Season."""
    df = df.copy()
    
    if fit:
        crop_encoder = LabelEncoder()
        state_encoder = LabelEncoder()
        df["Crop_encoded"] = crop_encoder.fit_transform(df["Crop"].astype(str))
        df["State_encoded"] = state_encoder.fit_transform(df["State"].astype(str))
    else:
        # Handle unseen labels
        crop_classes = set(crop_encoder.classes_)
        state_classes = set(state_encoder.classes_)
        df["Crop_encoded"] = df["Crop"].apply(
            lambda x: crop_encoder.transform([x])[0] if x in crop_classes else -1
        )
        df["State_encoded"] = df["State"].apply(
            lambda x: state_encoder.transform([x])[0] if x in state_classes else -1
        )
    
    season_dummies = pd.get_dummies(df["Season"], prefix="Season")
    df = pd.concat([df, season_dummies], axis=1)
    
    df = df.drop(columns=["Crop", "Season", "State"])
    return df, crop_encoder, state_encoder


def create_features(df):
    """Step 2.2 - Create engineered features. Exclude Production_per_Area to avoid leakage."""
    df = df.copy()
    df["Fertilizer_per_Area"] = df["Fertilizer"] / (df["Area"] + 1)
    df["Pesticide_per_Area"] = df["Pesticide"] / (df["Area"] + 1)
    df["Rainfall_Fertilizer_Ratio"] = df["Annual_Rainfall"] / (df["Fertilizer"] + 1)
    # Do NOT add Production_per_Area - it equals Yield (data leakage)
    return df


def get_feature_columns(df, exclude_cols=None):
    """Get list of feature columns (exclude target and non-features)."""
    exclude = {"Yield", "Production"} | (exclude_cols or set())
    return [c for c in df.columns if c not in exclude]


def run_feature_engineering(df, fit_encoders=True, crop_enc=None, state_enc=None, scaler=None):
    crop_enc = crop_enc
    state_enc = state_enc
    
    # Encode
    df, crop_enc, state_enc = encode_categoricals(
        df, crop_enc, state_enc, fit=fit_encoders
    )
    
    # Drop Production (leakage - Production/Area = Yield)
    df = df.drop(columns=["Production"], errors="ignore")
    
    # Create features
    df = create_features(df)
    
    feature_cols = get_feature_columns(df)
    return df, feature_cols, crop_enc, state_enc
