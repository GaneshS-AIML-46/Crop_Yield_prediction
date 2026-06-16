# Crop Yield Prediction

**Internship Project** — Made by Ganesh S

[![Live Demo](https://img.shields.io/badge/Live_Demo-GitHub_Pages-2d5a27?style=for-the-badge)](https://YOUR_USERNAME.github.io/crop-yield-prediction/)
[![API](https://img.shields.io/badge/API-Render-46a3ff?style=for-the-badge)](https://crop-yield-api.onrender.com/health)

End-to-end machine learning project: data cleaning → feature engineering → LightGBM model → FastAPI API → responsive web UI → free deployment on **GitHub Pages + Render**.

---

## Live Links (update after deploy)

| Link | URL |
|------|-----|
| **Live Demo** | `https://YOUR_USERNAME.github.io/crop-yield-prediction/` |
| **Source Code** | `https://github.com/YOUR_USERNAME/crop-yield-prediction` |
| **API Health** | `https://YOUR_RENDER_URL.onrender.com/health` |

> First prediction may take 30–60 seconds (Render free tier cold start).

---

## Tech Stack

| Layer | Technology | Why |
|-------|------------|-----|
| Language | Python 3 | ML + API in one ecosystem |
| Data | Pandas, NumPy, SciPy | Cleaning, transforms, stats |
| ML | scikit-learn, LightGBM | Preprocessing + regression (LightGBM = small model for GitHub) |
| API | FastAPI, Uvicorn, Pydantic | Fast REST API with validation |
| Frontend | HTML, CSS, JavaScript | Cross-platform (mobile + web), no build step |
| Hosting | GitHub Pages + Render | Free tier for portfolio showcase |

---

## Features

- Predict crop yield (tons/hectare) from crop, season, state, area, rainfall, fertilizer, pesticide
- Data pipeline: whitespace fix, outlier capping per crop, feature engineering
- LightGBM model with log-transform target (Test R² ~0.86 on original scale)
- REST API: `/predict`, `/crops`, `/states`, `/health`
- Mobile-friendly responsive UI

---

## Quick Start (Local)

```bash
cd cropYeild
python -m venv venv
venv\Scripts\activate          # Windows
pip install -r requirements.txt

python scripts/data_cleaning.py
python scripts/train_model.py

uvicorn api.main:app --reload --host 0.0.0.0
# In another terminal:
python -m http.server 3000 -d frontend
# Open http://localhost:3000
```

---

## Project Structure

```
cropYeild/
├── crop_yield.csv              # Raw dataset (~19k rows)
├── data/crop_yield_cleaned.csv
├── models/                     # LightGBM + encoders (~300 KB model)
├── scripts/                    # Cleaning, features, training
├── api/main.py                 # FastAPI backend
├── frontend/                   # GitHub Pages site
│   ├── index.html
│   └── config.js               # API URL (local vs production)
├── render.yaml                 # Render.com deploy config
└── requirements.txt
```

---

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/health` | Health check |
| GET | `/crops` | Valid crop names |
| GET | `/states` | Valid state names |
| GET | `/seasons` | Valid seasons |
| POST | `/predict` | Predict yield (JSON body) |

---

## Deployment (Free)

1. **GitHub** — Push this repo (public)
2. **Render** — Deploy backend from repo (`render.yaml` included)
3. **GitHub Pages** — Settings → Pages → Branch `main`, folder `/frontend`
4. **Update** `frontend/config.js` → set `PRODUCTION_API_URL` to your Render URL

See [DEPLOYMENT.md](DEPLOYMENT.md) for step-by-step instructions.  
Quick push guide: [GITHUB_PUSH.md](GITHUB_PUSH.md)

---

## Model Performance

| Metric | Value |
|--------|-------|
| Model | LightGBM |
| Test R² (original scale) | ~0.86 |
| Train–Test gap | < 0.02 |
| Model file size | ~0.3 MB (GitHub-safe) |

Production column excluded from features to prevent data leakage.

---

## Author

**Ganesh S** — Internship Project

For detailed architecture and design decisions, see [DOCUMENTATION.md](DOCUMENTATION.md).
