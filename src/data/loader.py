"""Data loading and initial cleaning."""

import pandas as pd


def load_data(filepath: str) -> pd.DataFrame:
    """Load raw telecom churn CSV.

    Expected columns: customerID, gender, SeniorCitizen, Partner,
    Dependents, tenure, PhoneService, MultipleLines, InternetService,
    OnlineSecurity, OnlineBackup, DeviceProtection, TechSupport,
    StreamingTV, StreamingMovies, Contract, PaperlessBilling,
    PaymentMethod, MonthlyCharges, TotalCharges, Churn
    """
    df = pd.read_csv(filepath)
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean the raw dataframe.

    - Drop customerID (no predictive value)
    - Convert TotalCharges to numeric (has empty strings)
    - Drop rows with NaN TotalCharges (~11 rows)
    - Reset index
    """
    df = df.copy()
    df = df.drop(columns=['customerID'], errors='ignore')
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df = df.dropna(subset=['TotalCharges'])
    df = df.reset_index(drop=True)
    return df


def split_features_target(df: pd.DataFrame, target: str = 'Churn'):
    """Split dataframe into X and y."""
    X = df.drop(columns=[target])
    y = (df[target] == 'Yes').astype(int)
    return X, y
