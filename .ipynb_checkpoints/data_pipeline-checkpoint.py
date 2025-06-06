# data_pipeline.py

import pandas as pd
import os
import logging
from sklearn.preprocessing import LabelEncoder

logging.basicConfig(level=logging.INFO)

REQUIRED_COLUMNS = [
    "Id", "Income", "Age", "Experience", "Married/Single",
    "House_Ownership", "Car_Ownership", "Profession", "CITY", "STATE",
    "CURRENT_JOB_YRS", "CURRENT_HOUSE_YRS"
]

def load_data(filepath: str) -> pd.DataFrame:
    """Load JSON data from the specified path."""
    logging.info(f"Loading data from {filepath}")
    
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"{filepath} not found.")

    try:
        df = pd.read_json(filepath)
    except ValueError as e:
        raise ValueError(f"Error reading JSON: {e}")

    validate_columns(df)
    return df

def validate_columns(df: pd.DataFrame) -> None:
    """Ensure all required columns are present."""
    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")
    logging.info("Column validation passed.")

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Perform basic data cleaning and encoding."""
    logging.info("Cleaning data...")

    # Handle missing values
    df.fillna(method="ffill", inplace=True)

    # Encode categorical columns
    categorical_cols = df.select_dtypes(include=["object"]).columns.tolist()
    for col in categorical_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))

    return df

def load_and_clean(filepath: str) -> pd.DataFrame:
    """Convenience function: load + validate + clean"""
    df = load_data(filepath)
    df = clean_data(df)
    return df
