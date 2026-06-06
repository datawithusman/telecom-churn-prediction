"""Visualization helpers."""

import matplotlib.pyplot as plt
import seaborn as sns


def plot_churn_distribution(df, target='Churn'):
    """Bar chart of churn distribution."""
    fig, ax = plt.subplots(figsize=(6, 4))
    df[target].value_counts().plot(kind='bar', ax=ax, color=['#2ecc71', '#e74c3c'])
    ax.set_title('Churn Distribution')
    ax.set_ylabel('Count')
    plt.tight_layout()
    return fig


def plot_correlation_heatmap(df, cols=None):
    """Heatmap of feature correlations."""
    numeric = df.select_dtypes(include='number') if cols is None else df[cols]
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(numeric.corr(), annot=True, fmt='.2f', cmap='coolwarm', ax=ax)
    ax.set_title('Feature Correlations')
    plt.tight_layout()
    return fig
