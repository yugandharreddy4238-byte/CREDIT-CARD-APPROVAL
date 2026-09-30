"""
ibm_watson_deploy.py
---------------------
Deploys the trained credit-card-approval model to IBM Watson Machine Learning
so it can be consumed as a scalable, real-time REST scoring endpoint (in
addition to / instead of the local Flask app).

Prerequisites
-------------
    pip install ibm-watson-machine-learning

    Set the following environment variables (or edit the CONFIG block below):
        WML_API_KEY       -> your IBM Cloud API key
        WML_URL           -> region URL, e.g. https://us-south.ml.cloud.ibm.com
        WML_SPACE_ID      -> target deployment space ID

Usage
-----
    python ibm_watson_deploy.py            # store + deploy the model
    python ibm_watson_deploy.py --score    # send a sample scoring request
"""

import argparse
import os
import sys
from pathlib import Path

import joblib

ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT / "models"
sys.path.insert(0, str(ROOT / "src"))

CONFIG = {
    "api_key": os.environ.get("WML_API_KEY", "<YOUR_IBM_CLOUD_API_KEY>"),
    "url": os.environ.get("WML_URL", "https://us-south.ml.cloud.ibm.com"),
    "space_id": os.environ.get("WML_SPACE_ID", "<YOUR_DEPLOYMENT_SPACE_ID>"),
}

MODEL_NAME = "credit-card-approval-model"
DEPLOYMENT_NAME = "credit-card-approval-deployment"


def get_client():
    from ibm_watson_machine_learning import APIClient

    wml_credentials = {"apikey": CONFIG["api_key"], "url": CONFIG["url"]}
    client = APIClient(wml_credentials)
    client.set.default_space(CONFIG["space_id"])
    return client


def store_and_deploy():
    client = get_client()

    model = joblib.load(MODELS_DIR / "best_model.pkl")
    metadata = joblib.load(MODELS_DIR / "metadata.pkl")

    # 1. Register the model with Watson Machine Learning ------------------
    sw_spec_id = client.software_specifications.get_id_by_name("runtime-23.1-py3.10")
    model_meta_props = {
        client.repository.ModelMetaNames.NAME: MODEL_NAME,
        client.repository.ModelMetaNames.TYPE: "scikit-learn_1.1",
        client.repository.ModelMetaNames.SOFTWARE_SPEC_ID: sw_spec_id,
    }
    stored_model_details = client.repository.store_model(
        model=model, meta_props=model_meta_props
    )
    model_id = client.repository.get_model_id(stored_model_details)
    print(f"Model stored. model_id={model_id} (best model: {metadata['best_model_name']})")

    # 2. Create an online deployment (scalable REST endpoint) -------------
    deployment_meta_props = {
        client.deployments.ConfigurationMetaNames.NAME: DEPLOYMENT_NAME,
        client.deployments.ConfigurationMetaNames.ONLINE: {},
    }
    deployment_details = client.deployments.create(model_id, meta_props=deployment_meta_props)
    deployment_id = client.deployments.get_id(deployment_details)
    scoring_url = client.deployments.get_scoring_href(deployment_details)

    print(f"Deployment created. deployment_id={deployment_id}")
    print(f"Scoring endpoint: {scoring_url}")
    print("\nCall it from any client with a payload like:")
    print(
        '  {"input_data": [{"fields": [...FEATURE_COLS...], '
        '"values": [[<row of encoded feature values>]]}]}'
    )
    return deployment_id


def score_sample(deployment_id: str):
    from preprocessing import FeatureEncoder
    import pandas as pd

    client = get_client()
    encoder = FeatureEncoder.load(MODELS_DIR / "feature_encoder.pkl")

    sample = pd.DataFrame([{
        "CNT_CHILDREN": 1, "AMT_INCOME_TOTAL": 220000, "AGE_YEARS": 34,
        "YEARS_EMPLOYED": 6, "FLAG_WORK_PHONE": 1, "FLAG_PHONE": 1,
        "FLAG_EMAIL": 1, "CNT_FAM_MEMBERS": 3, "CODE_GENDER": "F",
        "FLAG_OWN_CAR": "Y", "FLAG_OWN_REALTY": "Y",
        "NAME_INCOME_TYPE": "Working", "NAME_EDUCATION_TYPE": "Higher education",
        "NAME_FAMILY_STATUS": "Married", "NAME_HOUSING_TYPE": "House / apartment",
        "OCCUPATION_TYPE": "Managers",
    }])
    X = encoder.transform(sample)

    payload = {
        client.deployments.ScoringMetaNames.INPUT_DATA: [
            {"fields": list(X.columns), "values": X.values.tolist()}
        ]
    }
    result = client.deployments.score(deployment_id, payload)
    print(result)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--score", metavar="DEPLOYMENT_ID", nargs="?", const=True)
    args = parser.parse_args()

    if args.score:
        dep_id = args.score if isinstance(args.score, str) else input("Deployment ID: ")
        score_sample(dep_id)
    else:
        store_and_deploy()
