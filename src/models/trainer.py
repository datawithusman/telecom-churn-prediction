"""Model training and evaluation."""

from sklearn.model_selection import cross_val_score
from sklearn.metrics import f1_score, roc_auc_score, recall_score, classification_report
import numpy as np


def evaluate_model(model, X_test, y_test) -> dict:
    """Evaluate model and return metrics dict."""
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else y_pred

    return {
        'f1': f1_score(y_test, y_pred),
        'roc_auc': roc_auc_score(y_test, y_proba),
        'recall': recall_score(y_test, y_pred),
        'report': classification_report(y_test, y_pred, output_dict=True),
    }


def cross_validate(model, X, y, cv=5) -> dict:
    """Run cross-validation and return mean scores."""
    f1 = cross_val_score(model, X, y, cv=cv, scoring='f1')
    auc = cross_val_score(model, X, y, cv=cv, scoring='roc_auc')
    return {
        'f1_mean': f1.mean(),
        'f1_std': f1.std(),
        'auc_mean': auc.mean(),
        'auc_std': auc.std(),
    }
