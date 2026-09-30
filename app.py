"""
app.py
------
Flask web application for real-time credit card approval prediction.

Loads the best model chosen by src/train_models.py (best_model.pkl) plus its
matching feature encoder (feature_encoder.pkl) and serves a simple, clean UI
where a credit analyst / applicant can submit a financial profile and get an
instant Approve / Reject prediction, along with the model's confidence.
"""

import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from flask import Flask, render_template, request

ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT / "models"
sys.path.insert(0, str(ROOT / "src"))  # so joblib can resolve FeatureEncoder class

app = Flask(__name__)

# --- Load artifacts once at startup ---------------------------------------
model = joblib.load(MODELS_DIR / "best_model.pkl")
encoder = joblib.load(MODELS_DIR / "feature_encoder.pkl")
metadata = joblib.load(MODELS_DIR / "metadata.pkl")
BEST_MODEL_NAME = metadata["best_model_name"]

GENDER_OPTIONS = ["M", "F"]
YES_NO_OPTIONS = ["Y", "N"]
INCOME_TYPE_OPTIONS = ["Working", "Commercial associate", "Pensioner", "State servant", "Student"]
EDUCATION_OPTIONS = [
    "Secondary / secondary special", "Higher education", "Incomplete higher",
    "Lower secondary", "Academic degree",
]
FAMILY_STATUS_OPTIONS = ["Married", "Single / not married", "Civil marriage", "Separated", "Widow"]
HOUSING_TYPE_OPTIONS = [
    "House / apartment", "With parents", "Municipal apartment",
    "Rented apartment", "Office apartment", "Co-op apartment",
]
OCCUPATION_OPTIONS = [
    "Laborers", "Core staff", "Sales staff", "Managers", "Drivers",
    "High skill tech staff", "Accountants", "Medicine staff",
    "Cooking staff", "Security staff", "Cleaning staff", "Other",
]

FORM_OPTIONS = dict(
    gender=GENDER_OPTIONS,
    yes_no=YES_NO_OPTIONS,
    income_type=INCOME_TYPE_OPTIONS,
    education=EDUCATION_OPTIONS,
    family_status=FAMILY_STATUS_OPTIONS,
    housing_type=HOUSING_TYPE_OPTIONS,
    occupation=OCCUPATION_OPTIONS,
)


def build_feature_row(form) -> pd.DataFrame:
    """Translate the raw HTML form into the exact row shape the encoder expects."""
    age_years = float(form["age_years"])
    years_employed = float(form["years_employed"])

    row = {
        "CNT_CHILDREN": int(form["cnt_children"]),
        "AMT_INCOME_TOTAL": float(form["income"]),
        "AGE_YEARS": age_years,
        "YEARS_EMPLOYED": years_employed,
        "FLAG_WORK_PHONE": int(form.get("flag_work_phone", 0)),
        "FLAG_PHONE": int(form.get("flag_phone", 0)),
        "FLAG_EMAIL": int(form.get("flag_email", 0)),
        "CNT_FAM_MEMBERS": int(form["cnt_fam_members"]),
        "CODE_GENDER": form["gender"],
        "FLAG_OWN_CAR": form["own_car"],
        "FLAG_OWN_REALTY": form["own_realty"],
        "NAME_INCOME_TYPE": form["income_type"],
        "NAME_EDUCATION_TYPE": form["education_type"],
        "NAME_FAMILY_STATUS": form["family_status"],
        "NAME_HOUSING_TYPE": form["housing_type"],
        "OCCUPATION_TYPE": form["occupation_type"],
    }
    return pd.DataFrame([row])


@app.route("/", methods=["GET"])
def index():
    return render_template("index.html", options=FORM_OPTIONS, model_name=BEST_MODEL_NAME)


@app.route("/predict", methods=["POST"])
def predict():
    try:
        raw_row = build_feature_row(request.form)
        X = encoder.transform(raw_row)

        pred = int(model.predict(X)[0])
        proba = model.predict_proba(X)[0]
        confidence = float(proba[pred]) * 100

        result = {
            "decision": "REJECTED" if pred == 1 else "APPROVED",
            "is_approved": pred == 0,
            "confidence": round(confidence, 1),
            "risk_probability": round(float(proba[1]) * 100, 1),
            "model_name": BEST_MODEL_NAME,
        }
        return render_template(
            "index.html", options=FORM_OPTIONS, model_name=BEST_MODEL_NAME,
            result=result, form_data=request.form,
        )
    except Exception as exc:  # noqa: BLE001
        error = f"Could not process this application: {exc}"
        return render_template(
            "index.html", options=FORM_OPTIONS, model_name=BEST_MODEL_NAME,
            error=error, form_data=request.form,
        )


@app.route("/api/predict", methods=["POST"])
def api_predict():
    """JSON API endpoint — e.g. for the analyst dashboard or batch screening tools."""
    data = request.get_json(force=True)
    raw_row = build_feature_row(data)
    X = encoder.transform(raw_row)

    pred = int(model.predict(X)[0])
    proba = model.predict_proba(X)[0]

    return {
        "decision": "REJECTED" if pred == 1 else "APPROVED",
        "confidence": round(float(proba[pred]) * 100, 1),
        "risk_probability": round(float(proba[1]) * 100, 1),
        "model_used": BEST_MODEL_NAME,
    }


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
