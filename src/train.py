import pandas as pd
import numpy as np
import os
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score

INPUT_PATH = "data/features.csv"
MODEL_PATH = "artifacts/model.pkl"
SCALER_PATH = "artifacts/scaler.pkl"


def load_data():
    print("Loading feature dataset...")
    return pd.read_csv(INPUT_PATH)


def prepare_data(df):
    print("Preparing features...")

    # Drop columns not useful for training
    drop_cols = ["Time"]
    df = df.drop(columns=drop_cols, errors="ignore")

    X = df.drop("Class", axis=1)
    y = df["Class"]

    return X, y


def train_model(X, y):

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Model
    model = LogisticRegression(max_iter=1000, class_weight="balanced")

    print("Training model...")
    model.fit(X_train_scaled, y_train)

    # Predictions
    preds = model.predict(X_test_scaled)
    probs = model.predict_proba(X_test_scaled)[:, 1]

    # Metrics
    print("\nClassification Report:")
    print(classification_report(y_test, preds))

    roc = roc_auc_score(y_test, probs)
    print("ROC-AUC:", roc)

    return model, scaler


def save_artifacts(model, scaler):
    os.makedirs("artifacts", exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)
    print("Model saved.")


if __name__ == "__main__":
    df = load_data()
    X, y = prepare_data(df)
    joblib.dump(X.columns.tolist(),"artifacts/feature_names.pkl")
    model, scaler = train_model(X, y)
    save_artifacts(model, scaler)
