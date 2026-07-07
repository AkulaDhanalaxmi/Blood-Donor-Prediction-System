"""
Trains the blood-donor availability classifier and saves everything the
Flask app needs at inference time: the fitted model, the scaler, the label
encoder for blood_group, and the exact list of one-hot city columns (so a
single incoming donor record can be transformed identically to training).
"""
import json

import joblib
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from xgboost import XGBClassifier

DATA_PATH = "/home/claude/work/deploy/model/blood_donor_dataset.csv"
OUT_DIR = "/home/claude/work/deploy/model"

df = pd.read_csv(DATA_PATH)

le_blood = LabelEncoder()
df["blood_group_enc"] = le_blood.fit_transform(df["blood_group"])

le_target = LabelEncoder()
df["availability_enc"] = le_target.fit_transform(df["availability"])  # No=0, Yes=1

city_dummies = pd.get_dummies(df["city"], prefix="city", drop_first=True)
city_columns = city_dummies.columns.tolist()

features = pd.concat(
    [
        df[
            [
                "age",
                "weight",
                "blood_group_enc",
                "last_donation_days",
                "donations_count",
                "haemoglobin",
            ]
        ],
        city_dummies,
    ],
    axis=1,
)
feature_columns = features.columns.tolist()
target = df["availability_enc"]

x_train, x_test, y_train, y_test = train_test_split(
    features, target, test_size=0.2, random_state=20, stratify=target
)

scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)

smote = SMOTE(random_state=20)
x_train_sm, y_train_sm = smote.fit_resample(x_train_scaled, y_train)

candidates = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42),
    "Gradient Boosting": GradientBoostingClassifier(random_state=42),
    "XGBoost": XGBClassifier(n_estimators=200, random_state=42, eval_metric="logloss"),
}

results = {}
best_name, best_model, best_acc = None, None, -1

for name, model in candidates.items():
    model.fit(x_train_sm, y_train_sm)
    preds = model.predict(x_test_scaled)
    acc = accuracy_score(y_test, preds)
    results[name] = round(acc, 4)
    print(f"\n=== {name} ===")
    print("Accuracy:", acc)
    print(classification_report(y_test, preds, target_names=le_target.classes_))
    if acc > best_acc:
        best_acc = acc
        best_name = name
        best_model = model

print("\nBEST MODEL:", best_name, "Accuracy:", best_acc)

joblib.dump(best_model, f"{OUT_DIR}/model.joblib")
joblib.dump(scaler, f"{OUT_DIR}/scaler.joblib")
joblib.dump(le_blood, f"{OUT_DIR}/le_blood.joblib")
joblib.dump(le_target, f"{OUT_DIR}/le_target.joblib")

meta = {
    "best_model": best_name,
    "accuracy": round(best_acc, 4),
    "all_results": results,
    "feature_columns": feature_columns,
    "city_columns": city_columns,
    "cities": sorted(df["city"].unique().tolist()),
    "blood_groups": le_blood.classes_.tolist(),
}
with open(f"{OUT_DIR}/meta.json", "w") as f:
    json.dump(meta, f, indent=2)

print("\nSaved model artifacts to", OUT_DIR)
print(json.dumps(meta, indent=2))
