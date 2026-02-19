import pandas as pd
import numpy as np
import os

INPUT_PATH = "data/processed.csv"
OUTPUT_PATH = "data/features.csv"


def load_data():
    print("Loading processed data...")
    return pd.read_csv(INPUT_PATH)


def create_features(df):
    print("Creating features...")

    # Convert Time to hours
    df["hour"] = (df["Time"] // 3600) % 24

    # Amount normalization
    df["amount_log"] = df["Amount"].apply(lambda x: 0 if x <= 0 else np.log1p(x))

    # Simulated behavioral features
    df["is_night"] = df["hour"].apply(lambda x: 1 if x >= 22 or x <= 5 else 0)

    # Rolling transaction count (velocity feature simulation)
    df["tx_count_last_10"] = df["Time"].rolling(window=10, min_periods=1).count()

    # Rolling mean amount (behavior baseline)
    df["avg_amount_last_10"] = df["Amount"].rolling(window=10, min_periods=1).mean()

    # Deviation from recent behavior
    df["amount_deviation"] = df["Amount"] - df["avg_amount_last_10"]

    return df


def save_features(df):
    print("Saving feature dataset...")
    os.makedirs("data", exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False)


if __name__ == "__main__":
    df = load_data()
    df = create_features(df)
    save_features(df)
    print("Feature engineering complete.")
