# Pulse — Blood Donor Availability Predictor

A web app version of the Blood Donor Prediction System notebook. Enter a
donor's profile (age, weight, city, blood group, days since last donation,
donation history, haemoglobin) and get a live prediction of whether they're
likely to be available to donate, with a confidence breakdown.

## What's inside

```
app.py                  Flask app — serves the UI and the /predict API
templates/index.html    Front-end page
static/style.css        Design system (Fraunces + Inter + IBM Plex Mono)
static/app.js           Form → /predict → renders result
model/
  generate_dataset.py   Recreates a synthetic training dataset
  train_model.py        Trains + compares 4 models, saves the best one
  blood_donor_dataset.csv
  model.joblib, scaler.joblib, le_blood.joblib, le_target.joblib, meta.json
requirements.txt
Procfile                For Render/Heroku-style platforms
render.yaml             One-click blueprint for Render
```

## ⚠️ About the training data

The original project zip only contained the **executed notebook**, not the
raw `blood_donor_dataset.csv` it was trained on. So this includes a
synthetic dataset (`model/generate_dataset.py`) built to match the same
columns, ranges, and a realistic eligibility rule (≥90 days since last
donation, haemoglobin ≥ 12.5 g/dL, healthy weight), so the app works
end-to-end out of the box.

**To use your real data instead:** put your CSV at
`model/blood_donor_dataset.csv` with columns
`age, weight, city, blood_group, last_donation_days, donations_count, haemoglobin, availability`,
then run:

```bash
python3 model/train_model.py
```

This retrains Logistic Regression, Random Forest, Gradient Boosting, and
XGBoost, compares their accuracy, and automatically saves whichever one wins
as the model the app serves.

Current model (on synthetic data): **Gradient Boosting, 86% test accuracy**.

## Run locally

```bash
python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
python3 app.py
```

Visit `http://localhost:5000`.

## Deploy it

### Option A — Render (recommended, free tier, easiest)

1. Push this folder to a GitHub repo.
2. Go to [render.com](https://render.com) → **New +** → **Web Service** →
   connect your repo. Render will detect `render.yaml` automatically.
3. Or set it manually:
   - **Build command:** `pip install -r requirements.txt`
   - **Start command:** `gunicorn app:app`
4. Deploy. You'll get a live URL like `https://blood-donor-predictor.onrender.com`.

### Option B — Railway

1. [railway.app](https://railway.app) → **New Project** → **Deploy from GitHub repo**.
2. Railway auto-detects Python + the `Procfile`. No extra config needed.
3. Deploy → it gives you a public URL.

### Option C — PythonAnywhere (simple, no Docker/build steps)

1. Upload this folder (or `git clone` it) in a PythonAnywhere Bash console.
2. `pip install -r requirements.txt` (in a virtualenv).
3. Create a new **Web app** → Flask → point the WSGI file at `app.py`'s `app` object.
4. Reload the web app.

### Option D — Any VPS / Docker

```bash
pip install -r requirements.txt
gunicorn --bind 0.0.0.0:8000 app:app
```
Put Nginx or Caddy in front of it for TLS.

## API

`POST /predict`

```json
{
  "age": 34,
  "weight": 70,
  "city": "Pune",
  "blood_group": "O+",
  "last_donation_days": 150,
  "donations_count": 8,
  "haemoglobin": 13.9
}
```

Response:

```json
{
  "prediction": "Yes",
  "available": true,
  "confidence": 0.89,
  "probabilities": { "Yes": 0.89, "No": 0.11 }
}
```

## Credits

Original ML exploration (notebook comparing 8+ algorithms with SMOTE and
PCA) by Akula Dhanalaxmi, B.Tech CSE, SR University. This repo turns that
notebook into a deployable web app.
