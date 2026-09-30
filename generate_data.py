"""
generate_data.py
-----------------
Generates a synthetic, realistic credit-card-application dataset that mirrors
the structure of the widely used "Credit Card Approval Prediction" data:

    1. application_record.csv  -> demographic / financial info per applicant
    2. credit_record.csv       -> monthly repayment status history per applicant

These two files are merged and feature-engineered in src/preprocessing.py,
exactly the way you would with the real Kaggle dataset. Swap these CSVs out
for the real files (same column names) with zero code changes.
"""

import numpy as np
import pandas as pd

RNG = np.random.default_rng(42)
N_APPLICANTS = 8000


def generate_application_record(n=N_APPLICANTS) -> pd.DataFrame:
    ids = np.arange(5000000, 5000000 + n)

    gender = RNG.choice(["M", "F"], size=n, p=[0.45, 0.55])
    own_car = RNG.choice(["Y", "N"], size=n, p=[0.4, 0.6])
    own_realty = RNG.choice(["Y", "N"], size=n, p=[0.55, 0.45])
    cnt_children = RNG.choice([0, 1, 2, 3, 4], size=n, p=[0.55, 0.2, 0.15, 0.07, 0.03])

    # Income correlates loosely with income type / education
    income_type = RNG.choice(
        ["Working", "Commercial associate", "Pensioner", "State servant", "Student"],
        size=n, p=[0.55, 0.20, 0.13, 0.10, 0.02]
    )
    education_type = RNG.choice(
        ["Secondary / secondary special", "Higher education", "Incomplete higher",
         "Lower secondary", "Academic degree"],
        size=n, p=[0.55, 0.30, 0.08, 0.05, 0.02]
    )
    family_status = RNG.choice(
        ["Married", "Single / not married", "Civil marriage", "Separated", "Widow"],
        size=n, p=[0.55, 0.20, 0.10, 0.08, 0.07]
    )
    housing_type = RNG.choice(
        ["House / apartment", "With parents", "Municipal apartment",
         "Rented apartment", "Office apartment", "Co-op apartment"],
        size=n, p=[0.75, 0.10, 0.06, 0.05, 0.02, 0.02]
    )

    base_income = {
        "Working": 180000, "Commercial associate": 220000, "Pensioner": 110000,
        "State servant": 160000, "Student": 60000
    }
    edu_bonus = {
        "Secondary / secondary special": 1.0, "Higher education": 1.35,
        "Incomplete higher": 1.1, "Lower secondary": 0.85, "Academic degree": 1.6
    }
    income = np.array([
        max(27000, RNG.normal(base_income[it] * edu_bonus[et], 45000))
        for it, et in zip(income_type, education_type)
    ]).round(0)

    # Age in days-before-today (negative, as in real dataset), 21-70 yrs
    age_years = RNG.integers(21, 70, size=n)
    days_birth = -(age_years * 365 + RNG.integers(0, 365, size=n))

    # Employment length: pensioners/students get the "365243" sentinel (means not employed)
    days_employed = np.empty(n, dtype=int)
    for i, it in enumerate(income_type):
        if it in ("Pensioner", "Student"):
            days_employed[i] = 365243  # real dataset's "no job" sentinel
        else:
            emp_years = RNG.integers(0, min(40, age_years[i] - 18) + 1)
            days_employed[i] = -(emp_years * 365 + RNG.integers(0, 365))

    flag_mobil = np.ones(n, dtype=int)
    flag_work_phone = RNG.choice([0, 1], size=n, p=[0.77, 0.23])
    flag_phone = RNG.choice([0, 1], size=n, p=[0.72, 0.28])
    flag_email = RNG.choice([0, 1], size=n, p=[0.9, 0.1])

    occupation_type = RNG.choice(
        ["Laborers", "Core staff", "Sales staff", "Managers", "Drivers",
         "High skill tech staff", "Accountants", "Medicine staff",
         "Cooking staff", "Security staff", "Cleaning staff", "Other"],
        size=n
    )
    cnt_fam_members = np.clip(cnt_children + RNG.choice([1, 2], size=n), 1, None)

    df = pd.DataFrame({
        "ID": ids,
        "CODE_GENDER": gender,
        "FLAG_OWN_CAR": own_car,
        "FLAG_OWN_REALTY": own_realty,
        "CNT_CHILDREN": cnt_children,
        "AMT_INCOME_TOTAL": income,
        "NAME_INCOME_TYPE": income_type,
        "NAME_EDUCATION_TYPE": education_type,
        "NAME_FAMILY_STATUS": family_status,
        "NAME_HOUSING_TYPE": housing_type,
        "DAYS_BIRTH": days_birth,
        "DAYS_EMPLOYED": days_employed,
        "FLAG_MOBIL": flag_mobil,
        "FLAG_WORK_PHONE": flag_work_phone,
        "FLAG_PHONE": flag_phone,
        "FLAG_EMAIL": flag_email,
        "OCCUPATION_TYPE": occupation_type,
        "CNT_FAM_MEMBERS": cnt_fam_members,
    })
    return df


def generate_credit_record(app_df: pd.DataFrame) -> pd.DataFrame:
    """
    For each applicant, simulate between 1 and 36 months of repayment history.
    STATUS codes (mirrors real dataset):
        0  -> 1-29 days past due
        1  -> 30-59 days past due
        2  -> 60-89 days past due
        3  -> 90-119 days past due
        4  -> 120-149 days past due
        5  -> overdue > 150 days / written off as bad debt
        C  -> paid off that month, no debt
        X  -> no loan that month
    """
    rows = []

    # --- Risk probability is a function of the applicant's own profile,
    # so the trained models have real, learnable signal (mirrors how real
    # credit risk correlates with income, employment stability, age, etc.)
    income = app_df["AMT_INCOME_TOTAL"].to_numpy()
    income_z = (income - income.mean()) / income.std()

    days_employed = app_df["DAYS_EMPLOYED"].to_numpy()
    years_employed = np.where(days_employed == 365243, 0, -days_employed / 365.25)

    age_years = -app_df["DAYS_BIRTH"].to_numpy() / 365.25
    n_children = app_df["CNT_CHILDREN"].to_numpy()
    own_realty = (app_df["FLAG_OWN_REALTY"].to_numpy() == "Y").astype(float)
    own_car = (app_df["FLAG_OWN_CAR"].to_numpy() == "Y").astype(float)
    income_type = app_df["NAME_INCOME_TYPE"].to_numpy()
    education_type = app_df["NAME_EDUCATION_TYPE"].to_numpy()

    edu_risk = np.array([
        {"Academic degree": -0.6, "Higher education": -0.4, "Incomplete higher": 0.0,
         "Secondary / secondary special": 0.2, "Lower secondary": 0.5}[e]
        for e in education_type
    ])
    income_type_risk = np.array([
        {"Pensioner": -0.2, "State servant": -0.3, "Working": 0.1,
         "Commercial associate": -0.1, "Student": 0.5}[t]
        for t in income_type
    ])

    risk_score = (
        -0.55 * income_z
        - 0.03 * np.clip(years_employed, 0, 20)
        - 0.015 * (age_years - 35)
        + 0.12 * n_children
        - 0.35 * own_realty
        - 0.15 * own_car
        + edu_risk
        + income_type_risk
        + RNG.normal(0, 0.9, size=len(app_df))  # irreducible noise
    )
    # Convert score -> probability of being a "risky" applicant (~15% base rate)
    risky_prob = 1 / (1 + np.exp(-(risk_score - 1.2)))
    risky_mask = RNG.random(len(app_df)) < risky_prob

    for idx, (applicant_id, is_risky) in enumerate(zip(app_df["ID"], risky_mask)):
        n_months = RNG.integers(6, 37)
        if is_risky:
            status_pool = RNG.choice(
                ["C", "X", "0", "1", "2", "3", "4", "5"],
                size=n_months, p=[0.30, 0.15, 0.20, 0.15, 0.08, 0.06, 0.04, 0.02]
            )
        else:
            status_pool = RNG.choice(
                ["C", "X", "0", "1", "2", "3", "4", "5"],
                size=n_months, p=[0.58, 0.28, 0.135, 0.003, 0.0008, 0.0007, 0.0004, 0.0001]
            )
        for m, status in enumerate(status_pool):
            rows.append((applicant_id, -m, status))

    return pd.DataFrame(rows, columns=["ID", "MONTHS_BALANCE", "STATUS"])


if __name__ == "__main__":
    app_df = generate_application_record()
    credit_df = generate_credit_record(app_df)

    app_df.to_csv("application_record.csv", index=False)
    credit_df.to_csv("credit_record.csv", index=False)

    print(f"application_record.csv -> {app_df.shape}")
    print(f"credit_record.csv      -> {credit_df.shape}")
