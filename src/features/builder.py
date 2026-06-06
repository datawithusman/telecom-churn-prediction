"""Feature engineering pipeline."""

import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


class FeatureEngineer(BaseEstimator, TransformerMixin):
    """Custom feature engineering transformer.

    Creates:
    - tenure_group: binned tenure (0-12, 12-24, 24-48, 48-60, 60+)
    - avg_charges: TotalCharges / tenure
    - has_internet: 1 if InternetService != No
    - num_services: count of add-on services
    """

    ADD_ONS = ['OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
               'TechSupport', 'StreamingTV', 'StreamingMovies']

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        X['tenure_group'] = pd.cut(X['tenure'],
            bins=[0, 12, 24, 48, 60, 100],
            labels=['0-12', '12-24', '24-48', '48-60', '60+'])
        X['avg_monthly_charges'] = np.where(X['tenure'] > 0,
            X['TotalCharges'] / X['tenure'], X['MonthlyCharges'])
        X['has_internet'] = (X['InternetService'] != 'No').astype(int)
        X['num_services'] = sum(
            (X[col] == 'Yes').astype(int) for col in self.ADD_ONS if col in X.columns
        )
        return X
