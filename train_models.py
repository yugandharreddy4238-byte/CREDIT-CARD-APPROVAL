"""
train_models.py
----------------
Trains four classification algorithms on the credit-card-approval dataset,
compares them on standard classification metrics, and persists the
best-performing model (+ its feature encoder) to disk for use by the Flask
web app.

Algorithms:
    1. Logistic Regression
    2. Random Forest
    3. XGBoost (Gradient Boosting)  -- falls back to sklearn's
       GradientBoostingClassifier automatically if the `xgboost` package
       is not installed in the current environment.
    4. Decision Tree
"""

from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")  # headless plotting
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay, RocCurveDisplay, accuracy_score, classification_report,
    f1_score, precision_score, recall_score, roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

from preprocessing import build_dataset

try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
MODELS_DIR = ROOT / "models"
MODELS_DIR.mkdir(exist_ok=True)


def get_models():
    models = {
        "Logistic Regression": LogisticRegression(
            max_iter=1000, class_weight="balanced", random_state=42
        ),
        "Decision Tree": DecisionTreeClassifier(
            max_depth=8, class_weight="balanced", random_state=42
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=300, max_depth=12, class_weight="balanced",
            random_state=42, n_jobs=-1
        ),
    }

    if XGBOOST_AVAILABLE:
        # scale_pos_weight compensates for class imbalance (good >> bad)
        models["XGBoost"] = XGBClassifier(
            n_estimators=300, max_depth=6, learning_rate=0.08,
            eval_metric="logloss", random_state=42, n_jobs=-1
        )
    else:
        print("[warn] xgboost not installed in this environment -> using "
              "GradientBoostingClassifier as a drop-in stand-in labelled 'XGBoost'. "
              "Install `xgboost` (see requirements.txt) to use the real library.")
        models["XGBoost"] = GradientBoostingClassifier(
            n_estimators=300, max_depth=4, learning_rate=0.08, random_state=42
        )

    return models


def evaluate(name, model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    metrics = {
        "Model": name,
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred, zero_division=0),
        "Recall": recall_score(y_test, y_pred, zero_division=0),
        "F1 Score": f1_score(y_test, y_pred, zero_division=0),
        "ROC AUC": roc_auc_score(y_test, y_proba),
    }
    return metrics, y_pred, y_proba


def main():
    print("Loading & engineering features ...")
    X, y, encoder = build_dataset(
        DATA_DIR / "application_record.csv", DATA_DIR / "credit_record.csv"
    )
    print(f"Dataset shape: {X.shape}, class balance:\n{y.value_counts(normalize=True).round(3)}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    models = get_models()
    results = []
    fitted_models = {}
    roc_curves = {}

    for name, model in models.items():
        print(f"\nTraining {name} ...")
        model.fit(X_train, y_train)
        metrics, y_pred, y_proba = evaluate(name, model, X_test, y_test)
        results.append(metrics)
        fitted_models[name] = model
        roc_curves[name] = (y_test, y_proba)
        print(pd.Series(metrics).drop("Model").round(4).to_string())

    results_df = pd.DataFrame(results).sort_values("ROC AUC", ascending=False)
    print("\n=== Model comparison (sorted by ROC AUC) ===")
    print(results_df.round(4).to_string(index=False))
    results_df.to_csv(MODELS_DIR / "model_comparison.csv", index=False)

    best_name = results_df.iloc[0]["Model"]
    best_model = fitted_models[best_name]
    print(f"\nBest model: {best_name}")

    # Full classification report for the winner
    y_pred = best_model.predict(X_test)
    print(f"\nClassification report for {best_name}:")
    print(classification_report(y_test, y_pred, target_names=["Approve (0)", "Reject (1)"]))

    # Persist best model + encoder + metadata
    joblib.dump(best_model, MODELS_DIR / "best_model.pkl")
    encoder.save(MODELS_DIR / "feature_encoder.pkl")
    joblib.dump(
        {"best_model_name": best_name, "feature_cols": encoder.feature_cols},
        MODELS_DIR / "metadata.pkl",
    )
    print(f"\nSaved best_model.pkl, feature_encoder.pkl, metadata.pkl -> {MODELS_DIR}")

    # --- Plots -----------------------------------------------------------
    fig, ax = plt.subplots(figsize=(7, 5))
    for name, (yt, yp) in roc_curves.items():
        RocCurveDisplay.from_predictions(yt, yp, name=name, ax=ax)
    ax.set_title("ROC Curves — All Models")
    fig.tight_layout()
    fig.savefig(MODELS_DIR / "roc_curves.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5, 4))
    ConfusionMatrixDisplay.from_estimator(
        best_model, X_test, y_test, display_labels=["Approve", "Reject"],
        cmap="Blues", ax=ax
    )
    ax.set_title(f"Confusion Matrix — {best_name}")
    fig.tight_layout()
    fig.savefig(MODELS_DIR / "confusion_matrix.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    results_df.set_index("Model")[["Accuracy", "Precision", "Recall", "F1 Score", "ROC AUC"]].plot(
        kind="bar", ax=ax
    )
    ax.set_title("Model Comparison")
    ax.set_ylabel("Score")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(MODELS_DIR / "model_comparison.png", dpi=150)
    plt.close(fig)

    print("Saved evaluation plots (roc_curves.png, confusion_matrix.png, model_comparison.png)")


if __name__ == "__main__":
    main()
