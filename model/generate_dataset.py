"""
Generates a synthetic blood-donor dataset that mirrors the schema found in the
original project notebook (blood_donor_executed.ipynb):

    age, weight, city, blood_group, last_donation_days,
    donations_count, haemoglobin, availability

The original raw CSV was not included in the uploaded project (only the
executed notebook was), so this script recreates a realistic dataset with the
same columns, value ranges, and a sensible eligibility rule so a real model
can be trained end-to-end. Swap this out for the real CSV any time by
pointing train_model.py at it instead.
"""
import numpy as np
import pandas as pd

np.random.seed(42)

N = 3000

cities = ["Bangalore", "Chennai", "Delhi", "Hyderabad", "Kolkata", "Mumbai", "Pune"]
blood_groups = ["A+", "A-", "B+", "B-", "AB+", "AB-", "O+", "O-"]
blood_group_weights = [0.34, 0.06, 0.30, 0.05, 0.08, 0.02, 0.13, 0.02]

age = np.random.randint(18, 66, N)
weight = np.random.randint(45, 100, N)
city = np.random.choice(cities, N)
blood_group = np.random.choice(blood_groups, N, p=blood_group_weights)
last_donation_days = np.random.randint(0, 400, N)
donations_count = np.random.poisson(6, N) + 1
haemoglobin = np.round(np.random.normal(13.5, 1.4, N), 1).clip(9.0, 18.0)

# Eligibility follows real-world blood donation rules, loosely:
#  - must be >= 90 days since last donation
#  - haemoglobin >= 12.5 g/dL
#  - healthy weight (>= 50kg)
#  - a bit of noise so the model has something real to learn
score = (
    (last_donation_days >= 90).astype(float) * 0.45
    + (haemoglobin >= 12.5).astype(float) * 0.35
    + (weight >= 50).astype(float) * 0.10
    + (donations_count >= 3).astype(float) * 0.10
)
noise = np.random.normal(0, 0.18, N)
prob_available = np.clip(score + noise, 0, 1)
availability = np.where(prob_available > 0.72, "Yes", "No")

df = pd.DataFrame(
    {
        "age": age,
        "weight": weight,
        "city": city,
        "blood_group": blood_group,
        "last_donation_days": last_donation_days,
        "donations_count": donations_count,
        "haemoglobin": haemoglobin,
        "availability": availability,
    }
)

out_path = "/home/claude/work/deploy/model/blood_donor_dataset.csv"
df.to_csv(out_path, index=False)
print("Saved:", out_path, df.shape)
print(df["availability"].value_counts())
