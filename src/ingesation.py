import pandas as pd
import os

RAW_DATA_PATH = "data/raw.csv"
PROCESSED_DATA_PATH = "data/processed.csv"

def load_data():
    print("Loading dataset...")
    data = pd.read_csv(RAW_DATA_PATH)
    print("Shape:", data.shape)
    print("columns:", data.columns)
    return data


def basic_cleaning(data):
    print("cleaning data...")
    data = data.drop_duplicates()
    data = data.dropna()
    return data 


def save_data(data):
    print("Saving processed data..")
    os.makedirs("data", exist_ok=True)
    data.to_csv(PROCESSED_DATA_PATH, index=False)



if __name__ == "__main__":
    df = load_data()
    df = basic_cleaning(df)
    save_data(df)
    print("Ingestion completed...")    