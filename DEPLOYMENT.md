# Step-by-Step Deployment Guide

**Internship Project** — Made by Ganesh S

This guide walks you through deploying the Crop Yield Prediction app: Backend on Render.com and Frontend on Netlify.

---

## Prerequisites

- [ ] GitHub account ([github.com](https://github.com))
- [ ] Render account ([render.com](https://render.com)) — sign up with GitHub
- [ ] Netlify account ([netlify.com](https://netlify.com)) — sign up with GitHub
- [ ] Git installed on your computer
- [ ] Project already trained (models saved in `models/` folder)

---

# Part A: Deploy Backend on Render.com

## Step 1: Push Your Project to GitHub

### 1.1 Initialize Git (if not already done)

Open terminal/PowerShell in your project folder:

```bash
cd d:\cropYeild
git init
```

### 1.2 Create a `.gitignore` file (if not exists)

Create `d:\cropYeild\.gitignore` with:

```
venv/
__pycache__/
*.pyc
.env
.DS_Store
*.log
```

### 1.3 Add and commit all files

```bash
git add .
git status
git commit -m "Initial commit - Crop Yield Prediction"
```

### 1.4 Create a new repository on GitHub

1. Go to [github.com/new](https://github.com/new)
2. Repository name: `crop-yield-prediction` (or any name)
3. Choose **Public**
4. Leave "Add a README" **unchecked**
5. Click **Create repository**

### 1.5 Push to GitHub

Copy the commands GitHub shows (replace with your username and repo name):

```bash
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/crop-yield-prediction.git
git push -u origin main
```

Enter your GitHub username and password (or Personal Access Token if 2FA is enabled).

---

## Step 2: Create Web Service on Render

### 2.1 Log in to Render

1. Go to [render.com](https://render.com)
2. Click **Sign Up** or **Log In** → **Log in with GitHub**
3. Authorize Render to access your GitHub account

### 2.2 Create a new Web Service

1. From the Render dashboard, click **New +** → **Web Service**
2. If prompted, click **Connect account** to link your GitHub
3. Find and select your `crop-yield-prediction` repository
4. Click **Connect**

### 2.3 Configure the Web Service

| Field | Value |
|-------|-------|
| **Name** | `crop-yield-api` |
| **Region** | Choose nearest (e.g. Oregon or Singapore) |
| **Branch** | `main` |
| **Root Directory** | Leave empty |
| **Runtime** | `Python 3` |
| **Build Command** | `pip install -r requirements.txt` |
| **Start Command** | `uvicorn api.main:app --host 0.0.0.0 --port $PORT` |

### 2.4 Instance type

- Select **Free** plan

### 2.5 Deploy

1. Click **Create Web Service**
2. Wait 2–5 minutes for the first build
3. When done, you’ll see a URL like: `https://crop-yield-api-xxxx.onrender.com`
4. Click the URL and add `/health` — you should see: `{"status":"ok",...}`
5. **Copy this URL** — you’ll use it for the frontend (e.g. `https://crop-yield-api-xxxx.onrender.com`)

---

## Step 3: Fix Backend Issues (if needed)

### If build fails

1. In Render, go to **Logs**
2. Check for errors (often missing packages or wrong Python version)
3. Ensure `requirements.txt` has all needed packages
4. Make sure `api/` folder and `models/` folder are in the repo

### If model files are missing

1. Confirm `models/` folder is committed:
   - `crop_yield_model.pkl`
   - `scaler.pkl`
   - `crop_encoder.pkl`
   - `state_encoder.pkl`
   - `feature_names.json`

2. If missing, run locally first:
   ```bash
   python scripts/data_cleaning.py
   python scripts/train_model.py
   ```
   Then add and push:
   ```bash
   git add models/
   git commit -m "Add model files"
   git push
   ```

3. In Render, trigger a new deploy: **Manual Deploy** → **Deploy latest commit**

---

# Part B: Deploy Frontend on GitHub Pages (Recommended)

## Step 4: Update API URL in Frontend

The frontend uses [`frontend/config.js`](frontend/config.js) to switch between local and production API automatically.

### 4.1 Edit `frontend/config.js`

1. Open `frontend/config.js`
2. After deploying backend on Render, set your Render URL:
   ```javascript
   const PRODUCTION_API_URL = 'https://crop-yield-api-xxxx.onrender.com';
   ```
3. Save, commit, and push:
   ```bash
   git add frontend/config.js
   git commit -m "Update production API URL"
   git push
   ```

Local development still uses `http://localhost:8000` automatically.

---

## Step 5: Enable GitHub Pages

1. Open your repo on GitHub
2. Go to **Settings** → **Pages**
3. Under **Build and deployment**:
   - **Source:** Deploy from a branch
   - **Branch:** `main`
   - **Folder:** `/frontend`
4. Click **Save**
5. Wait 1–2 minutes for deployment
6. Your live URL: `https://YOUR_USERNAME.github.io/REPO_NAME/`

---

## Step 6: Test GitHub Pages Deployment

1. Open your GitHub Pages URL on desktop and mobile
2. Crop and State dropdowns should load from Render API
3. Submit a prediction (allow 30–60s on first request for cold start)
4. Verify result displays correctly

---

# Part C: Deploy Frontend on Netlify (Alternative)

## Step 4 (Netlify): Update API URL

Same as Part B — edit `frontend/config.js` with your Render URL.

## Step 5 (Netlify): Deploy on Netlify

### 5.1 Log in to Netlify

1. Go to [netlify.com](https://netlify.com)
2. Click **Sign up** or **Log in** → **Log in with GitHub**
3. Authorize Netlify to access your GitHub account

### 5.2 Add new site

1. Click **Add new site** → **Import an existing project**
2. Click **Deploy with GitHub**
3. Choose **GitHub** if asked for a provider
4. Authorize Netlify if prompted
5. You’ll see a list of your repositories

### 5.3 Select your repository

1. Find `crop-yield-prediction` (or your repo name)
2. Click **Select** or **Import**

### 5.4 Configure build settings

| Setting | Value |
|---------|-------|
| **Branch to deploy** | `main` |
| **Build command** | Leave empty (or type: `echo "Static site"`) |
| **Publish directory** | `frontend` |

⚠️ **Important:** `Publish directory` must be exactly `frontend` (the folder containing index.html).

### 5.5 Deploy

1. Click **Deploy site**
2. Wait 1–2 minutes
3. When finished, you’ll see: **Site is live**
4. Your URL will look like: `https://random-words-12345.netlify.app`
5. Click the URL to open the app

---

## Step 6: Test the Deployment

### 6.1 Test frontend

1. Open your Netlify URL
2. Dropdowns for Crop and State should load (from API)
3. Fill the form and click **Predict Yield**
4. On the first request, wait 30–60 seconds (Render free tier cold start)
5. You should see the predicted yield

### 6.2 If API calls fail

1. Open browser Developer Tools (F12) → Network tab
2. Reload and submit again
3. Check for CORS or 404 errors
4. Ensure `API_BASE_URL` in `index.html` matches your Render URL exactly (no trailing slash)

---

## Step 7: (Optional) Custom domain

### 7.1 Change site name on Netlify

1. Netlify dashboard → **Site settings** → **Change site name**
2. Choose something like `crop-yield-ganesh` → Save
3. New URL: `https://crop-yield-ganesh.netlify.app`

### 7.2 Add your own domain (optional)

1. **Site settings** → **Domain management** → **Add custom domain**
2. Follow Netlify’s instructions to point your domain’s DNS to Netlify

---

## Quick reference: URLs

| Service | URL example |
|---------|-------------|
| **Backend (Render)** | `https://crop-yield-api-xxxx.onrender.com` |
| **Frontend (GitHub Pages)** | `https://YOUR_USERNAME.github.io/crop-yield-prediction/` |
| **Frontend (Netlify)** | `https://your-site-name.netlify.app` |
| **Health check** | `https://your-backend.onrender.com/health` |

---

## Troubleshooting

### "Could not load crops/states"

- Backend may be starting (Render cold start). Wait 30–60 seconds and retry.
- Check that `PRODUCTION_API_URL` in `frontend/config.js` is correct.

### CORS errors in browser

- Your FastAPI app already enables CORS (`allow_origins=["*"]`).
- If issues persist, ensure you’re using the Netlify URL, not `file://`.

### 404 on API endpoints

- Confirm the Render service is running.
- Check Render logs for startup errors.

### Build fails on Netlify

- Ensure `Publish directory` is `frontend`.
- Ensure `frontend/index.html` exists in the repo.

---

## Summary checklist

- [ ] Push project to GitHub
- [ ] Create Web Service on Render
- [ ] Backend deployed and `/health` returns OK
- [ ] Update `PRODUCTION_API_URL` in `frontend/config.js`
- [ ] Enable GitHub Pages (branch `main`, folder `/frontend`)
- [ ] Test predict flow on live site
