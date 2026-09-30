"""
preprocessing.py
-----------------
Feature engineering pipeline for the Credit Card Approval project.

Key responsibilities:
1. Merge application_record.csv (applicant demographics/financials) with
   credit_record.csv (monthly repayment history).
2. Convert the multi-class STATUS payment codes (0,1,2,3,4,5,C,X) into a
   BINARY target label:
       TARGET = 1  -> "bad" client / high risk  -> reject
       TARGET = 0  -> "good" client / low risk  -> approve
   A client is labelled "bad" if they were ever 60+ days past due
   (STATUS in {2,3,4,5}), which mirrors standard credit-risk practice.
3. Engineer human-readable features (AGE_YEARS, YEARS_EMPLOYED, etc.)
4. Encode categoricals and scale numerics, returning a model-ready matrix.
"""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

# STATUS codes that indicate a seriously delinquent (high-risk) applicant
HIGH_RISK_STATUSES = {"2", "3", "4", "5"}

CATEGORICAL_COLS = [
    "CODE_GENDER", "FLAG_OWN_CAR", "FLAG_OWN_REALTY", "NAME_INCOME_TYPE",
    "NAME_EDUCATION_TYPE", "NAME_FAMILY_STATUS", "NAME_HOUSING_TYPE",
    "OCCUPATION_TYPE",
]

NUMERIC_COLS = [
    "CNT_CHILDREN", "AMT_INCOME_TOTAL", "AGE_YEARS", "YEARS_EMPLOYED",
    "FLAG_WORK_PHONE", "FLAG_PHONE", "FLAG_EMAIL", "CNT_FAM_MEMBERS",
]

FEATURE_COLS = NUMERIC_COLS + CATEGORICAL_COLS
TARGET_COL = "TARGET"


def build_binary_target(credit_df: pd.DataFrame) -> pd.DataFrame:
    """
    Collapse the multi-class STATUS payment codes per applicant into a single
    binary TARGET column (1 = high risk / reject, 0 = low risk / approve).
    """
    credit_df = credit_df.copy()
    credit_df["IS_HIGH_RISK_MONTH"] = credit_df["STATUS"].astype(str).isin(HIGH_RISK_STATUSES)

    risk_flag = (
        credit_df.groupby("ID")["IS_HIGH_RISK_MONTH"]
        .any()
        .astype(int)
        .rename(TARGET_COL)
        .reset_index()
    )
    return risk_flag


def engineer_features(app_df: pd.DataFrame) -> pd.DataFrame:
    df = app_df.copy()

    df["AGE_YEARS"] = (-df["DAYS_BIRTH"] / 365.25).round(1)

    # DAYS_EMPLOYED uses a 365243 sentinel for "not currently employed"
    df["YEARS_EMPLOYED"] = np.where(
        df["DAYS_EMPLOYED"] == 365243, 0, (-df["DAYS_EMPLOYED"] / 365.25).round(1)
    )
    df["YEARS_EMPLOYED"] = df["YEARS_EMPLOYED"].clip(lower=0)

    df["OCCUPATION_TYPE"] = df["OCCUPATION_TYPE"].fillna("Unknown")
    return df


def load_and_merge(app_path: str, credit_path: str) -> pd.DataFrame:
    app_df = pd.read_csv(app_path)
    credit_df = pd.read_csv(credit_path)

    app_df = engineer_features(app_df)
    target_df = build_binary_target(credit_df)

    merged = app_df.merge(target_df, on="ID", how="inner")
    return merged


class FeatureEncoder:
    """Fits label encoders + a scaler on training data; reusable at inference."""

    def __init__(self):
        self.encoders: dict[str, LabelEncoder] = {}
        self.scaler = StandardScaler()
        self.categorical_cols = CATEGORICAL_COLS
        self.numeric_cols = NUMERIC_COLS
        self.feature_cols = FEATURE_COLS

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        for col in self.categorical_cols:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            self.encoders[col] = le

        df[self.numeric_cols] = self.scaler.fit_transform(df[self.numeric_cols])
        return df[self.feature_cols]

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        for col in self.categorical_cols:
            le = self.encoders[col]
            # Gracefully handle unseen categories at inference time
            df[col] = df[col].astype(str).map(
                lambda v, le=le: v if v in le.classes_ else le.classes_[0]
            )
            df[col] = le.transform(df[col])

        df[self.numeric_cols] = self.scaler.transform(df[self.numeric_cols])
        return df[self.feature_cols]

    def save(self, path: str):
        joblib.dump(self, path)

    @staticmethod
    def load(path: str) -> "FeatureEncoder":
        return joblib.load(path)


def build_dataset(app_path: str, credit_path: str):
    """Convenience one-shot: returns (X, y, fitted encoder)."""
    merged = load_and_merge(app_path, credit_path)
    y = merged[TARGET_COL]
    encoder = FeatureEncoder()
    X = encoder.fit_transform(merged)
    return X, y, encoder


if __name__ == "__main__":
    here = Path(__file__).resolve().parent.parent / "data"
    X, y, encoder = build_dataset(here / "application_record.csv", here / "credit_record.csv")
    print("Feature matrix shape:", X.shape)
    print("Target distribution:\n", y.value_counts(normalize=True).round(3))
