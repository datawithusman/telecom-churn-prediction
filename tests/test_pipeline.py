"""Tests for the churn prediction pipeline."""

import pandas as pd
import numpy as np
from src.data.loader import clean_data, split_features_target
from src.features.builder import FeatureEngineer


def _make_sample_data(n=100):
    np.random.seed(42)
    return pd.DataFrame({
        'customerID': range(n),
        'gender': np.random.choice(['Male', 'Female'], n),
        'SeniorCitizen': np.random.choice([0, 1], n),
        'Partner': np.random.choice(['Yes', 'No'], n),
        'Dependents': np.random.choice(['Yes', 'No'], n),
        'tenure': np.random.randint(0, 72, n),
        'PhoneService': np.random.choice(['Yes', 'No'], n),
        'MultipleLines': np.random.choice(['Yes', 'No', 'No phone service'], n),
        'InternetService': np.random.choice(['DSL', 'Fiber optic', 'No'], n),
        'OnlineSecurity': np.random.choice(['Yes', 'No', 'No internet service'], n),
        'OnlineBackup': np.random.choice(['Yes', 'No', 'No internet service'], n),
        'DeviceProtection': np.random.choice(['Yes', 'No', 'No internet service'], n),
        'TechSupport': np.random.choice(['Yes', 'No', 'No internet service'], n),
        'StreamingTV': np.random.choice(['Yes', 'No', 'No internet service'], n),
        'StreamingMovies': np.random.choice(['Yes', 'No', 'No internet service'], n),
        'Contract': np.random.choice(['Month-to-month', 'One year', 'Two year'], n),
        'PaperlessBilling': np.random.choice(['Yes', 'No'], n),
        'PaymentMethod': np.random.choice(['Electronic check', 'Mailed check', 'Bank transfer', 'Credit card'], n),
        'MonthlyCharges': np.random.uniform(18, 120, n).round(2),
        'TotalCharges': np.random.uniform(18, 8000, n).round(2),
        'Churn': np.random.choice(['Yes', 'No'], n),
    })


def test_clean_data():
    df = _make_sample_data()
    cleaned = clean_data(df)
    assert 'customerID' not in cleaned.columns
    assert cleaned['TotalCharges'].dtype in ['float64', 'int64']
    assert cleaned.isnull().sum().sum() == 0


def test_split_features_target():
    df = clean_data(_make_sample_data())
    X, y = split_features_target(df)
    assert 'Churn' not in X.columns
    assert set(y.unique()) == {0, 1}


def test_feature_engineer():
    df = clean_data(_make_sample_data())
    X, _ = split_features_target(df)
    fe = FeatureEngineer()
    X_eng = fe.fit_transform(X)
    assert 'tenure_group' in X_eng.columns
    assert 'avg_monthly_charges' in X_eng.columns
    assert 'num_services' in X_eng.columns
    assert 'has_internet' in X_eng.columns
