# Telecom Customer Churn Prediction

End-to-end ML pipeline predicting which telecom customers will churn. Built as a portfolio project demonstrating the full ML workflow: EDA, preprocessing, feature engineering, model selection, and deployment.

## Results

| Model | F1 Score | ROC-AUC | Recall |
|-------|----------|---------|--------|
| Logistic Regression | 0.60 | 0.84 | 0.55 |
| Random Forest | 0.56 | 0.82 | 0.48 |
| **XGBoost** | **0.65** | **0.85** | **0.60** |
| XGBoost + SMOTE | 0.63 | 0.84 | 0.68 |

XGBoost won on F1/AUC. SMOTE variant trades precision for recall (catches more churners).

## Project Structure

```
telecom-churn-prediction/
├── data/               # Raw + processed data (.gitignored)
├── notebooks/          # EDA + experiments
├── src/
│   ├── data/           # Load + clean
│   ├── features/       # Feature engineering
│   ├── models/         # Train + evaluate
│   └── viz/            # Plots + charts
├── models/             # Saved model artifacts
├── tests/
├── requirements.txt
└── train.py            # End-to-end pipeline
```

## Quickstart

```bash
git clone https://github.com/yourusername/telecom-churn-prediction.git
cd telecom-churn-prediction
pip install -r requirements.txt

# Download dataset from Kaggle
kaggle datasets download -d blastchar/telco-customer-churn
unzip telco-customer-churn.zip -d data/raw/

# Run the full pipeline
python train.py
```

## Key Decisions

1. **Why XGBoost over Random Forest?** — Gradient boosting handles class imbalance better, and the dataset had non-linear feature interactions (tenure × contract type).
2. **Why not SMOTE by default?** — It boosted recall but tanked precision. For churn, false positives (offering retention discount to someone who wasn't leaving) cost money too. Default model optimizes F1.
3. **Feature importance** — Month-to-month contracts, fiber optic internet, and tenure < 12 months are the top 3 churn drivers. Makes business sense.

## What I Learned

- Class imbalance is hard. Tried class weights, SMOTE, and threshold tuning — no silver bullet.
- SHAP values > feature importance for explaining individual predictions to stakeholders.
- Calibration matters: the raw XGBoost probabilities overestimate churn for low-risk customers.