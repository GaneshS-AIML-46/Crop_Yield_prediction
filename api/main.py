"""
Phase 5 — FastAPI Backend for Crop Yield Prediction
FIX: log1p target transform — predictions are inverse-transformed via expm1.
"""
import json
import joblib
import numpy as np
from pathlib import Path
from collections import OrderedDict
from typing import Optional

import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "models"

# In-memory cache (Phase 8)
MODEL_CACHE = {}
PREDICTION_HISTORY: OrderedDict = OrderedDict()
MAX_HISTORY = 100

# Valid options
SEASON_OPTIONS = ["Autumn", "Kharif", "Rabi", "Summer", "Whole Year", "Winter"]


class PredictRequest(BaseModel):
    crop: str = Field(..., min_length=1)
    crop_year: int = Field(..., ge=1990, le=2030)
    season: str = Field(..., min_length=1)
    state: str = Field(..., min_length=1)
    area: float = Field(..., gt=0)
    annual_rainfall: float = Field(..., ge=0)
    fertilizer: float = Field(..., ge=0)
    pesticide: float = Field(..., ge=0)


class PredictResponse(BaseModel):
    predicted_yield: float
    unit: str = "tons/hectare"
    confidence: str


def load_artifacts():
    """Load model and encoders once at startup."""
    if MODEL_CACHE:
        return
    required = ["crop_yield_model.pkl", "scaler.pkl", "crop_encoder.pkl", "state_encoder.pkl", "feature_names.json"]
    for f in required:
        p = MODELS_DIR / f
        if not p.exists():
            raise FileNotFoundError(f"Model artifact not found: {p}")
    MODEL_CACHE["model"]        = joblib.load(MODELS_DIR / "crop_yield_model.pkl")
    MODEL_CACHE["scaler"]       = joblib.load(MODELS_DIR / "scaler.pkl")
    MODEL_CACHE["crop_encoder"] = joblib.load(MODELS_DIR / "crop_encoder.pkl")
    MODEL_CACHE["state_encoder"] = joblib.load(MODELS_DIR / "state_encoder.pkl")
    with open(MODELS_DIR / "feature_names.json") as f:
        MODEL_CACHE["feature_names"] = json.load(f)
    MODEL_CACHE["crops"]  = list(MODEL_CACHE["crop_encoder"].classes_)
    MODEL_CACHE["states"] = list(MODEL_CACHE["state_encoder"].classes_)
    # Load transform config (log1p fix)
    config_path = MODELS_DIR / "model_config.json"
    if config_path.exists():
        with open(config_path) as f:
            cfg = json.load(f)
        MODEL_CACHE["log_transform"] = cfg.get("log_transform", False)
    else:
        MODEL_CACHE["log_transform"] = False


def preprocess_input(req: PredictRequest) -> pd.DataFrame:
    """Apply same preprocessing as training pipeline."""
    crop = str(req.crop).strip()
    season = str(req.season).strip()
    state = str(req.state).strip()

    crop_enc = MODEL_CACHE["crop_encoder"]
    state_enc = MODEL_CACHE["state_encoder"]
    feature_names = MODEL_CACHE["feature_names"]

    if crop not in crop_enc.classes_:
        raise HTTPException(status_code=400, detail=f"Unknown crop: {crop}. Use GET /crops for valid options.")
    if state not in state_enc.classes_:
        raise HTTPException(status_code=400, detail=f"Unknown state: {state}. Use GET /states for valid options.")
    if season not in SEASON_OPTIONS:
        raise HTTPException(status_code=400, detail=f"Invalid season: {season}. Use one of {SEASON_OPTIONS}")

    crop_encoded = crop_enc.transform([crop])[0]
    state_encoded = state_enc.transform([state])[0]

    # Build one-hot season (ensure all Season_* columns exist)
    season_cols = [c for c in feature_names if c.startswith("Season_")]
    season_row = {c: 0 for c in season_cols}
    key = f"Season_{season}"
    if key in season_row:
        season_row[key] = 1

    row = {
        "Crop_Year": req.crop_year,
        "Area": req.area,
        "Annual_Rainfall": req.annual_rainfall,
        "Fertilizer": req.fertilizer,
        "Pesticide": req.pesticide,
        "Crop_encoded": crop_encoded,
        "State_encoded": state_encoded,
        "Fertilizer_per_Area": req.fertilizer / (req.area + 1),
        "Pesticide_per_Area": req.pesticide / (req.area + 1),
        "Rainfall_Fertilizer_Ratio": req.annual_rainfall / (req.fertilizer + 1),
        **season_row,
    }
    df = pd.DataFrame([row])
    df = df[feature_names]
    return df


def get_confidence(pred: float) -> str:
    """Simple confidence based on prediction magnitude."""
    if pred < 0:
        return "low"
    if pred > 100:
        return "medium"
    return "high"


app = FastAPI(title="Crop Yield Prediction API", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def startup():
    load_artifacts()


@app.get("/health")
def health():
    return {"status": "ok", "message": "Crop Yield API is running"}


@app.get("/crops")
def get_crops():
    load_artifacts()
    return {"crops": MODEL_CACHE["crops"]}


@app.get("/states")
def get_states():
    load_artifacts()
    return {"states": MODEL_CACHE["states"]}


@app.get("/seasons")
def get_seasons():
    return {"seasons": SEASON_OPTIONS}


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    load_artifacts()
    df = preprocess_input(req)
    scaler = MODEL_CACHE["scaler"]
    model  = MODEL_CACHE["model"]
    X    = scaler.transform(df)
    pred = float(model.predict(X)[0])
    # Inverse-transform if model was trained on log1p(Yield)
    if MODEL_CACHE.get("log_transform", False):
        pred = float(np.expm1(pred))
    pred = max(0.0, pred)

    # LRU cache for last 100 predictions
    key = f"{req.crop}|{req.season}|{req.state}|{req.crop_year}"
    PREDICTION_HISTORY[key] = {"input": req.model_dump(), "predicted_yield": pred}
    if len(PREDICTION_HISTORY) > MAX_HISTORY:
        PREDICTION_HISTORY.popitem(last=False)

    return PredictResponse(
        predicted_yield=round(pred, 4),
        confidence=get_confidence(pred),
    )
