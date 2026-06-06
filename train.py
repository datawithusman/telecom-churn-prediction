"""End-to-end training pipeline for telecom churn prediction."""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
import joblib
import os

from src.data.loader import load_data, clean_data, split_features_target
from src.features.builder import FeatureEngineer
from src.models.trainer import evaluate_model


def build_pipeline(model):
    """Build preprocessing + model pipeline."""
    categorical = ['gender', 'Partner', 'Dependents', 'PhoneService',
                   'MultipleLines', 'InternetService', 'OnlineSecurity',
                   'OnlineBackup', 'DeviceProtection', 'TechSupport',
                   'StreamingTV', 'StreamingMovies', 'Contract',
                   'PaperlessBilling', 'PaymentMethod', 'tenure_group']
    numeric = ['tenure', 'MonthlyCharges', 'TotalCharges',
               'avg_monthly_charges', 'num_services']

    preprocessor = ColumnTransformer([
        ('cat', OneHotEncoder(drop='first', handle_unknown='ignore'), categorical),
        ('num', StandardScaler(), numeric),
    ])

    return Pipeline([
        ('features', FeatureEngineer()),
        ('preprocess', preprocessor),
        ('model', model),
    ])


def main():
    DATA_PATH = os.path.join("data", "raw", "WA_Fn_UseC_Telco_Customer_Churn.csv")
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(
            "Dataset not found. Run: kaggle datasets download -d blastchar/telco-customer-chunk"
        )

    print('Loading data...')
    df = load_data(DATA_PATH)
    df = clean_data(df)
    X, y = split_features_target(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f'Train: {len(X_train)}, Test: {len(X_test)}')
    print(f'Churn rate (train): {y_train.mean():.2%}')

    models = {
        'logistic': LogisticRegression(max_iter=1000, random_state=42),
        'random_forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'xgboost': XGBClassifier(n_estimators=200, max_depth=5, learning_rate=0.1,
                                  random_state=42, eval_metric='logloss'),
    }

    results = {}
    best_f1 = 0
    best_name = None
    best_model = None

    for name, model in models.items():
        print(f'\nTraining {name}...')
        pipe = build_pipeline(model)
        pipe.fit(X_train, y_train)
        metrics = evaluate_model(pipe, X_test, y_test)
        results[name] = metrics
        print(f'  F1: {metrics["f1"]:.4f}  AUC: {metrics["roc_auc"]:.4f}  Recall: {metrics["recall"]:.4f}')

        if metrics['f1'] > best_f1:
            best_f1 = metrics['f1']
            best_name = name
            best_model = pipe

    print(f'\nBest model: {best_name} (F1={best_f1:.4f})')

    os.makedirs('models', exist_ok=True)
    joblib.dump(best_model, f'models/{best_name}_churn.joblib')
    print(f'Saved to models/{best_name}_churn.joblib')


if __name__ == "__main__":
    main()
