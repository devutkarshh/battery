"""
Verify the browser model.js conversion reproduces sklearn predictions.

Run (from repo root):
    py web/scripts/verify_model.py
"""
import json
import os
import pickle

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
MODEL_JS = os.path.join(ROOT, "web", "js", "model.js")
PKL = os.path.join(ROOT, "outputs", "battery_rul_rf_model.pkl")
DATASET = os.path.join(ROOT, "Battery_dataset.csv")

pkg = pickle.load(open(PKL, "rb"))
rf = pkg["model"]
features = pkg["features"]

# load the exported JS (strip the window assignment and leading comment)
txt = "\n".join(
    ln for ln in open(MODEL_JS).read().splitlines() if not ln.strip().startswith("/*")
)
obj = json.loads(txt.strip().rstrip(";").split("=", 1)[1].strip())

df = pd.read_csv(DATASET)
df = df.sort_values(["battery_id", "cycle"]).reset_index(drop=True)

for bid in df["battery_id"].unique():
    mask = df["battery_id"] == bid
    df.loc[mask, "capacity_fade_rate"] = df.loc[mask, "BCt"].diff().fillna(0)
    df.loc[mask, "SOH_rate"] = df.loc[mask, "SOH"].diff().fillna(0)
    df.loc[mask, "temp_rise_rate"] = df.loc[mask, "chT"].diff().fillna(0)
    df.loc[mask, "voltage_drop"] = df.loc[mask, "chV"] - df.loc[mask, "disV"]
    df.loc[mask, "BCt_rolling_mean"] = df.loc[mask, "BCt"].rolling(5, min_periods=1).mean()
    df.loc[mask, "SOH_rolling_mean"] = df.loc[mask, "SOH"].rolling(5, min_periods=1).mean()


def rf_predict(x):
    """Browser-equivalent inference (pure numpy) from the exported trees."""
    vals = np.zeros(len(obj["trees"]))
    for i, t in enumerate(obj["trees"]):
        node = 0
        while t["left"][node] != -1:
            node = t["left"][node] if x[t["feature"][node]] <= t["threshold"][node] else t["right"][node]
        vals[i] = t["value"][node]
    return vals.mean()


X = df[features]
y = df["RUL"]
truth = rf.predict(X)

max_err = 0.0
worst = None
for i in range(len(X)):
    pred = rf_predict(X.iloc[i].to_numpy())
    err = abs(pred - truth[i])
    if err > max_err:
        max_err = err
        worst = (i, pred, truth[i])

print(f"Compared {len(X)} rows")
print(f"Max |js_export - sklearn| error = {max_err:.4f} cycles  (row {worst[0]} -> js {worst[1]:.2f} vs sklearn {worst[2]:.2f})")
print("PASS" if max_err < 0.01 else "FAIL")