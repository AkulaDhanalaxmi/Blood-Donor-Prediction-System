import json
import os

import joblib
import numpy as np
import pandas as pd
from flask import Flask, jsonify, render_template, request

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "model")

app = Flask(__name__)

model = joblib.load(os.path.join(MODEL_DIR, "model.joblib"))
scaler = joblib.load(os.path.join(MODEL_DIR, "scaler.joblib"))
le_blood = joblib.load(os.path.join(MODEL_DIR, "le_blood.joblib"))
le_target = joblib.load(os.path.join(MODEL_DIR, "le_target.joblib"))

with open(os.path.join(MODEL_DIR, "meta.json")) as f:
    META = json.load(f)

FEATURE_COLUMNS = META["feature_columns"]
CITY_COLUMNS = META["city_columns"]
CITIES = META["cities"]
BLOOD_GROUPS = META["blood_groups"]


def build_feature_row(payload):
    age = float(payload["age"])
    weight = float(payload["weight"])
    city = payload["city"]
    blood_group = payload["blood_group"]
    last_donation_days = float(payload["last_donation_days"])
    donations_count = float(payload["donations_count"])
    haemoglobin = float(payload["haemoglobin"])

    blood_group_enc = le_blood.transform([blood_group])[0]

    row = {
        "age": age,
        "weight": weight,
        "blood_group_enc": blood_group_enc,
        "last_donation_days": last_donation_days,
        "donations_count": donations_count,
        "haemoglobin": haemoglobin,
    }
    for col in CITY_COLUMNS:
        row[col] = 0
    city_col = f"city_{city}"
    if city_col in row:
        row[city_col] = 1
    # If city == the dropped baseline category (first alphabetically), all
    # dummy columns stay 0, which is the correct encoding.

    df = pd.DataFrame([row])[FEATURE_COLUMNS]
    return df


@app.route("/")
def index():
    return render_template(
        "index.html",
        cities=CITIES,
        blood_groups=BLOOD_GROUPS,
        model_name=META["best_model"],
        accuracy=META["accuracy"],
    )


@app.route("/api/meta")
def api_meta():
    return jsonify(META)


@app.route("/predict", methods=["POST"])
def predict():
    payload = request.get_json(force=True)

    required = [
        "age",
        "weight",
        "city",
        "blood_group",
        "last_donation_days",
        "donations_count",
        "haemoglobin",
    ]
    missing = [k for k in required if k not in payload or payload[k] in ("", None)]
    if missing:
        return jsonify({"error": f"Missing fields: {', '.join(missing)}"}), 400

    try:
        row = build_feature_row(payload)
    except Exception as exc:  # noqa: BLE001
        return jsonify({"error": f"Invalid input: {exc}"}), 400

    scaled = scaler.transform(row)
    pred_enc = model.predict(scaled)[0]
    proba = model.predict_proba(scaled)[0]
    classes = le_target.classes_.tolist()
    prob_map = {cls: round(float(p), 4) for cls, p in zip(classes, proba)}
    label = le_target.inverse_transform([pred_enc])[0]

    return jsonify(
        {
            "prediction": label,
            "available": bool(label == "Yes"),
            "probabilities": prob_map,
            "confidence": round(float(max(proba)), 4),
        }
    )


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
