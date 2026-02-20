from fastapi import FastAPI
import joblib
import numpy as np
import pandas as pd
import logging
import os

os.makedirs("logs", exist_ok=True)


# Logging setup
logging.basicConfig(
    filename="logs/predictions.log",
    level=logging.INFO,
    format="%(asctime)s - %(message)s"
)

app = FastAPI(title="Fraud Detection API")

# Load artifacts
model = joblib.load("artifacts/model.pkl")
scaler = joblib.load("artifacts/scaler.pkl")
feature_names = joblib.load("artifacts/feature_names.pkl")


@app.get("/")
def home():
    return {"message": "Fraud Detection API Running"}


@app.post("/predict")
def predict(features: dict):
    try:
        df = pd.DataFrame([features])

        for col in feature_names:
            if col not in df.columns:
                df[col] = 0

        df = df[feature_names]

        scaled = scaler.transform(df)
        
        prob = model.predict_proba(scaled)[0][1]

        risk = "HIGH" if prob > 0.7 else "MEDIUM" if prob > 0.4 else "LOW"

        logging.info(
            f"INPUT={features} | FRAUD_PROB={prob} | RISK={risk}"
        )

        return {
            "fraud_probability": float(prob),
            "risk_level": risk
        }

    except Exception as e:
        logging.error(str(e))
        return {"error": str(e)}
