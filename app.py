"""
Flask web app for the measles outbreak predictor.
Loads trained models and serves predictions via API endpoints.
"""
import os
import pickle
import pandas as pd
import numpy as np
from flask import Flask, render_template, jsonify

from src.features import build_features, FEATURE_COLS

app = Flask(__name__)

# ── load data and models at startup ──────────────────────────────────────────

def load_everything():
    df_raw = pd.read_csv("data/measles_weekly.csv", parse_dates=["date"])
    df = build_features(df_raw)

    with open("data/model_reg.pkl", "rb") as f:
        reg_model = pickle.load(f)
    with open("data/model_clf.pkl", "rb") as f:
        clf_model = pickle.load(f)

    return df, reg_model, clf_model

try:
    DF, REG_MODEL, CLF_MODEL = load_everything()
    READY = True
except Exception as e:
    print(f"Warning: could not load models — run src/train.py first. ({e})")
    READY = False


RISK_LABELS = {0: "Low", 1: "Medium", 2: "High"}
RISK_COLORS = {0: "#3b6d11", 1: "#b45309", 2: "#a32d2d"}


def make_forecast():
    """Use the last available row to forecast the next 4 weeks."""
    last = DF.dropna(subset=FEATURE_COLS).iloc[-1]
    X = last[FEATURE_COLS].values.reshape(1, -1)

    predicted_cases = max(0, int(REG_MODEL.predict(X)[0]))
    risk_idx = int(CLF_MODEL.predict(X)[0])
    risk_proba = CLF_MODEL.predict_proba(X)[0]

    return {
        "predicted_cases_4w": predicted_cases,
        "risk": RISK_LABELS[risk_idx],
        "risk_color": RISK_COLORS[risk_idx],
        "risk_proba": {
            "Low": round(float(risk_proba[0]) * 100, 1),
            "Medium": round(float(risk_proba[1]) * 100, 1),
            "High": round(float(risk_proba[2]) * 100, 1),
        },
        "as_of": str(DF["date"].max().date()),
    }


# ── routes ────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/historical")
def historical():
    """Monthly aggregated cases for the trend chart."""
    df = DF.copy()
    df["month"] = df["date"].dt.to_period("M").dt.to_timestamp()
    monthly = df.groupby("month")["cases"].sum().reset_index()
    return jsonify({
        "labels": monthly["month"].dt.strftime("%b %Y").tolist(),
        "values": monthly["cases"].tolist(),
    })


@app.route("/api/forecast")
def forecast():
    if not READY:
        return jsonify({"error": "Models not loaded. Run src/train.py first."}), 503
    return jsonify(make_forecast())


@app.route("/api/summary")
def summary():
    df = DF.copy()
    total_2025 = int(df[df["year"] == 2025]["cases"].sum())
    total_2026 = int(df[df["year"] == 2026]["cases"].sum())
    latest_4w = int(df.tail(4)["cases"].sum())
    avg_vax = round(float(df.iloc[-1]["vax_rate"]), 1)
    return jsonify({
        "total_2025": total_2025,
        "total_2026": total_2026,
        "latest_4w_cases": latest_4w,
        "vax_rate": avg_vax,
    })


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=True, host="0.0.0.0", port=port)
