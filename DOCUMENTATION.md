# Crop Yield Prediction — Technical Documentation

**Internship Project** — Made by Ganesh S

> End-to-end technical documentation covering the tech stack, rationale for each choice, and a detailed explanation of the entire pipeline from raw data to deployed application.

---

## Table of Contents

1. [Tech Stack Overview](#tech-stack-overview)
2. [Why Each Technology Was Chosen](#why-each-technology-was-chosen)
3. [End-to-End Pipeline](#end-to-end-pipeline)
4. [Architecture & Data Flow](#architecture--data-flow)
5. [Design Decisions](#design-decisions)
6. [File Structure & Responsibilities](#file-structure--responsibilities)

---

## Tech Stack Overview

| Layer | Technology | Purpose |
|-------|------------|---------|
| **Language** | Python 3.x | Data science ecosystem, ML libraries, API development |
| **Data Processing** | Pandas, NumPy | Load, clean, transform tabular data |
| **ML / Modeling** | scikit-learn | Preprocessing, baseline & tree models, evaluation |
| **Model Persistence** | Joblib | Save/load trained models and encoders |
| **API Backend** | FastAPI | REST API for predictions |
| **ASGI Server** | Uvicorn | Run FastAPI in production |
| **Validation** | Pydantic | Request validation, type safety |
| **Frontend** | HTML, CSS, JavaScript (vanilla) | Single-page form UI, no framework |
| **Deployment (Backend)** | Render.com | Host FastAPI on free tier |
| **Deployment (Frontend)** | Netlify | Static hosting for HTML/CSS/JS |
| **Scientific Computing** | SciPy | Statistical utilities (e.g., winsorization) |

---

## Why Each Technology Was Chosen

### Python

**What it is:** A general-purpose programming language with strong support for data science and web development.

**Why used:**  
- Dominant language for machine learning (scikit-learn, pandas, NumPy are Python-first).  
- Single language for data cleaning, model training, and API development.  
- Large ecosystem and community for agricultural/regression use cases.

---

### Pandas

**What it is:** A library for structured data manipulation (DataFrames) and analysis.

**Why used:**  
- CSV loading and inspection.  
- Group-based imputation (e.g., median by Crop + Season).  
- Easy handling of missing values, duplicates, and type conversion.  
- Native integration with NumPy and scikit-learn.

---

### NumPy

**What it is:** A library for numerical arrays and mathematical operations.

**Why used:**  
- Underpins pandas and scikit-learn.  
- Percentile/IQR calculations for outlier treatment.  
- Efficient array operations in feature engineering (e.g., clipping, ratios).

---

### scikit-learn

**What it is:** Machine learning library for preprocessing, models, and evaluation.

**Why used:**  
- **Preprocessing:** `LabelEncoder`, `StandardScaler`, `train_test_split`.  
- **Models:** Linear Regression (baseline), Random Forest, Gradient Boosting.  
- **Evaluation:** RMSE, MAE, R².  
- **Search:** RandomizedSearchCV for hyperparameter tuning.  
- Stable API and production-ready implementation.

---

### SciPy

**What it is:** Scientific computing library built on NumPy.

**Why used:**  
- Statistical utilities (e.g., winsorization) for outlier treatment.  
- Standard dependency for scientific/ML Python projects.

---

### Joblib

**What it is:** Lightweight serialization for Python objects, especially NumPy arrays.

**Why used:**  
- Recommended by scikit-learn for saving/loading models.  
- Handles large NumPy arrays and complex objects (sklearn pipelines, encoders).  
- Faster and more suitable than pickle for ML artifacts.

---

### FastAPI

**What it is:** Modern, async-capable web framework for building APIs.

**Why used:**  
- High performance (Starlette + Pydantic).  
- Automatic OpenAPI docs and interactive UI.  
- Native Pydantic integration for validation and type hints.  
- Straightforward setup and deployment.  
- Lower latency than frameworks like Flask for similar use cases.

---

### Uvicorn

**What it is:** ASGI server for running async Python web apps.

**Why used:**  
- Reference server for FastAPI.  
- Supports async and production-style workers.  
- Easy to configure host/port for Render and local development.  
- Handles concurrent requests efficiently.

---

### Pydantic

**What it is:** Data validation library using Python type hints.

**Why used:**  
- Request body validation (e.g., `crop_year` 1990–2030, `area > 0`).  
- Clear error messages for invalid inputs.  
- Type conversion (string → int/float).  
- Integrated with FastAPI for automatic validation.

---

### Vanilla HTML / CSS / JavaScript

**What it is:** Frontend built with no frameworks (no React, Vue, etc.).

**Why used:**  
- No build step or bundler.  
- Works on any host (Netlify, GitHub Pages).  
- Small footprint and fast load.  
- Sufficient for a form and single prediction result.

---

### Render.com (Backend)

**What it is:** Cloud platform for hosting web services and APIs.

**Why used:**  
- Free tier for web services.  
- Native support for Python and `requirements.txt`.  
- Simple deployment via Git.  
- Built-in health checks for the API.

---

### Netlify (Frontend)

**What it is:** Platform for hosting static sites.

**Why used:**  
- Free tier for static hosting.  
- Deploy from Git with minimal configuration.  
- Global CDN and HTTPS.  
- Works well for HTML/CSS/JS only.

---

## End-to-End Pipeline

### Stage 1: Raw Data

- **Source:** `crop_yield.csv`  
- **Size:** ~19,689 rows, 10 columns  
- **Columns:** Crop, Crop_Year, Season, State, Area, Production, Annual_Rainfall, Fertilizer, Pesticide, Yield  
- **Target:** `Yield` (continuous regression)

---

### Stage 2: Data Cleaning & Preprocessing

| Step | Action | Reason |
|------|--------|--------|
| **Inspection** | Shape, dtypes, nulls, duplicates, basic stats | Understand data quality and anomalies |
| **Whitespace** | Strip leading/trailing spaces from Crop, Season, State | Raw data has "Kharif     ", "Whole Year ", etc. |
| **Missing values** | Drop rows where Yield is null; fill Area, Production with median by Crop+Season; fill Rainfall, Fertilizer, Pesticide with median by State+Crop_Year | Preserve structure and avoid bias |
| **Outliers** | Cap Yield at 1st–99th percentile per Crop | Different crops have different scales; global removal would wrongly trim Coconut/Sugarcane |
| **Duplicates** | Remove rows with same Crop, Crop_Year, Season, State | Avoid duplicate influence in training |
| **Dtypes** | Crop_Year → int; numeric cols → float64; string cols → string | Consistent types for encoding and scaling |

Output: `data/crop_yield_cleaned.csv`

---

### Stage 3: Feature Engineering

| Step | Action | Reason |
|------|--------|--------|
| **Label encode Crop** | Map ~55 crops to integers | Too many categories for one-hot; tree models handle labels well |
| **Label encode State** | Map ~30 states to integers | Same rationale as Crop |
| **One-hot encode Season** | 6 binary columns | Only 6 values; each season is a distinct category |
| **Drop Production** | Remove column | Avoid leakage: Production/Area ≈ Yield |
| **Engineered features** | Fertilizer_per_Area, Pesticide_per_Area, Rainfall_Fertilizer_Ratio | Domain-informed ratios improve signal |
| **Feature selection** | Keep features with low multicollinearity | Reduce noise and redundancy |

Output: Feature matrix `X` and target vector `y`.

---

### Stage 4: Train / Validation / Test Split

- **Test:** 20% of data  
- **Remaining 80%:** 80/20 split into train and validation  
- **Final:** 64% train, 16% validation, 20% test  
- **Random state:** 42 for reproducibility  
- **Shuffle:** Enabled for random splits

---

### Stage 5: Scaling

- **Method:** StandardScaler (zero mean, unit variance)  
- **Fit:** On training data only  
- **Transform:** Train, validation, and test  
- **Excluded:** One-hot Season columns (already 0/1)

Avoids fitting the scaler on the full dataset to prevent data leakage.

---

### Stage 6: Model Training

| Phase | Action |
|-------|--------|
| **Baseline** | Linear Regression — establishes minimum performance |
| **Comparison** | Random Forest, Gradient Boosting, XGBoost (if installed), LightGBM (if installed) |
| **Selection** | Best model by validation R² |
| **Optional tuning** | RandomizedSearchCV on best model (can be enabled via flag) |

Output: `models/crop_yield_model.pkl`, plus scaler and encoders.

---

### Stage 7: API Backend

1. **Startup:** Load model, scaler, crop/state encoders, feature names.  
2. **Request:** JSON with crop, crop_year, season, state, area, annual_rainfall, fertilizer, pesticide.  
3. **Preprocessing:** Same as training: encode, engineer features, scale.  
4. **Prediction:** Model predicts Yield.  
5. **Response:** JSON with predicted_yield, unit, and a simple confidence flag.

---

### Stage 8: Frontend

1. **Load:** Fetch crops and states from `/crops` and `/states`.  
2. **Form:** User fills dropdowns and number inputs.  
3. **Submit:** POST to `/predict`.  
4. **Result:** Display predicted yield, gauge, and input summary.  
5. **Errors:** Show clear messages for validation or API failures.

---

### Stage 9: Deployment

- **Backend:** Push to Git → Render builds from `requirements.txt` and runs `uvicorn api.main:app`.  
- **Frontend:** Push to Git → Netlify deploys the `frontend` folder.  
- **CORS:** Enabled in FastAPI so the frontend domain can call the API.  
- **API URL:** Frontend uses `API_BASE_URL` pointing to the Render backend.

---

## Architecture & Data Flow

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           CROP YIELD PREDICTION                              │
└─────────────────────────────────────────────────────────────────────────────┘

  Raw Data (crop_yield.csv)
           │
           ▼
  ┌─────────────────────┐
  │  Data Cleaning      │  pandas, NumPy, SciPy
  │  (scripts/)         │  • Whitespace strip
  │                     │  • Missing value imputation
  │                     │  • Per-crop outlier capping
  │                     │  • Duplicate removal
  └──────────┬──────────┘
             │
             ▼
  ┌─────────────────────┐
  │  Feature Eng.       │  pandas, scikit-learn
  │  (scripts/)         │  • LabelEncode Crop, State
  │                     │  • One-hot Season
  │                     │  • Engineered features
  │                     │  • Drop Production (leakage)
  └──────────┬──────────┘
             │
             ▼
  ┌─────────────────────┐
  │  Train/Val/Test     │  scikit-learn
  │  StandardScaler     │  • 64/16/20 split
  │  Model Training     │  • Fit scaler on train only
  └──────────┬──────────┘
             │
             ▼
  ┌─────────────────────┐
  │  Persist Artifacts  │  joblib, json
  │  (models/)          │  • model.pkl, scaler.pkl
  │                     │  • crop_encoder, state_encoder
  │                     │  • feature_names.json
  └──────────┬──────────┘
             │
             ▼
  ┌─────────────────────┐     HTTP GET/POST      ┌─────────────────────┐
  │  FastAPI Backend    │◄──────────────────────►│  Frontend (HTML)    │
  │  (api/main.py)      │                        │  (frontend/)        │
  │  • /predict         │     JSON responses     │  • Form + Fetch     │
  │  • /crops, /states  │                        │  • Vanilla JS       │
  │  • /health          │                        └──────────┬──────────┘
  └──────────┬──────────┘                                   │
             │                                              │
             ▼                                              ▼
  ┌─────────────────────┐                        ┌─────────────────────┐
  │  Render.com         │                        │  Netlify            │
  │  (Backend Host)     │                        │  (Static Host)      │
  └─────────────────────┘                        └─────────────────────┘
```

---

## Design Decisions

### 1. Per-Crop Outlier Treatment

Different crops have very different yield scales (e.g., Coconut vs Rice). Global IQR-based removal would discard valid Coconut data. We cap at 1st–99th percentile **per Crop** instead.

### 2. Excluding Production from Features

Yield is defined as Production / Area. Including Production or Production_per_Area would leak the target. We exclude both from features.

### 3. Label vs One-Hot Encoding

- **Crop, State:** Label encoding to avoid high-dimensional one-hot for ~55 and ~30 values.  
- **Season:** One-hot encoding because there are only 6 values and they represent distinct categories.

### 4. StandardScaler Only on Numeric Features

One-hot Season columns are already 0/1. Scaling them does not add value; scaling is applied only to numeric features.

### 5. Fit Scaler on Train Only

The scaler is fit on the training set and reused for validation and test. This reflects real deployment: the scaler is trained once and applied to new data.

### 6. In-Memory Model Loading

Model, scaler, and encoders are loaded once at startup and stored in memory. This avoids disk reads on every prediction and keeps latency low.

### 7. LRU Prediction Cache

The last 100 predictions (by input key) are cached in memory. This supports basic debugging and future enhancements without adding database dependencies.

### 8. Vanilla Frontend

No React/Vue/Angular to keep dependencies minimal, build simple, and hosting straightforward on GitHub Pages.

---

## File Structure & Responsibilities

| Path | Purpose |
|------|---------|
| `crop_yield.csv` | Raw dataset |
| `data/crop_yield_cleaned.csv` | Cleaned dataset |
| `scripts/data_cleaning.py` | Cleaning pipeline (whitespace, nulls, outliers, duplicates, dtypes) |
| `scripts/feature_engineering.py` | Encoding and engineered features |
| `scripts/train_model.py` | Split, scale, train, evaluate, save artifacts |
| `models/crop_yield_model.pkl` | Trained LightGBM model |
| `models/scaler.pkl` | StandardScaler fitted on train |
| `models/crop_encoder.pkl` | LabelEncoder for Crop |
| `models/state_encoder.pkl` | LabelEncoder for State |
| `models/feature_names.json` | Ordered feature names for preprocessing |
| `models/model_config.json` | Log-transform and model metadata |
| `api/main.py` | FastAPI app: load artifacts, preprocess, predict, CORS |
| `frontend/index.html` | Single-page form UI and API calls |
| `frontend/config.js` | API URL (local vs production) |
| `requirements.txt` | Python dependencies |
| `Procfile` | Render start command |
| `render.yaml` | Render service config |
| `README.md` | Quick start and usage |
| `DOCUMENTATION.md` | This technical documentation |
| `DEPLOYMENT.md` | Step-by-step deploy guide |

---

*This documentation reflects the project as implemented. For updates or customizations, refer to the source code and comments in the respective files.*
