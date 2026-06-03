<div align="center">

# 📊 Telecom Customer Churn Prediction

**Machine Learning model that identifies at-risk customers before they leave — with actionable business insights**

[![Python](https://img.shields.io/badge/Python-3776AB?style=flat-square&logo=python&logoColor=white)]()
[![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=flat-square&logo=scikit-learn&logoColor=white)]()
[![pandas](https://img.shields.io/badge/pandas-150458?style=flat-square&logo=pandas&logoColor=white)]()
[![matplotlib](https://img.shields.io/badge/matplotlib-11557C?style=flat-square)]()
[![seaborn](https://img.shields.io/badge/seaborn-3776AB?style=flat-square)]()
[![Jupyter](https://img.shields.io/badge/Jupyter-F37626?style=flat-square&logo=jupyter&logoColor=white)]()

*It costs 5-7x more to acquire a new customer than to keep an existing one. This model helps telecom companies stop churn before it happens.*

**[📓 Kaggle Notebook](https://kaggle.com/datawithusman)** · **[📂 Dataset](#dataset)**

</div>

---

## 🎯 Why This Matters

Telecom companies face **1.5-2% monthly churn rates**, translating to millions in lost revenue. This project demonstrates how to:

- Predict which customers are likely to churn with **~80% accuracy**
- Identify the **top drivers** of customer attrition
- Translate model findings into **actionable retention strategies**

---

## 📊 Dataset

The dataset contains **7,043 customer records** from a telecom company with the following features:

| Feature Category | Examples | Count |
|-----------------|----------|-------|
| **Demographics** | Gender, Senior Citizen, Dependents | 3 |
| **Services** | Phone, Internet, Streaming, Tech Support | 9 |
| **Account** | Tenure, Contract, Payment Method, Charges | 5 |
| **Target** | Churn (Yes/No) | 1 |

**Class Distribution:**
- ✅ Retained: 73.5% (5,174 customers)
- ❌ Churned: 26.5% (1,869 customers)

---

## 🧠 Models & Results

| Model | Accuracy | Precision | Recall | F1-Score |
|-------|----------|-----------|--------|----------|
| Logistic Regression | 0.80 | 0.79 | 0.80 | 0.79 |
| Random Forest | 0.78 | 0.77 | 0.78 | 0.77 |
| Decision Tree | 0.73 | 0.72 | 0.73 | 0.72 |
| KNN | 0.76 | 0.75 | 0.76 | 0.75 |

> **Best Model: Logistic Regression** — selected for the best balance of performance and interpretability, with class balancing applied via SMOTE.

---

## 🔑 Key Insights

### Top Churn Drivers

```
Feature Importance (Logistic Regression Coefficients)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 tenure (months)          ████████████████████  -0.35  (lower = higher risk)
 contract_type_monthly    ██████████████████    +0.32  (month-to-month churns most)
 internet_service_fiber   ███████████████       +0.28  (surprisingly high churn)
 monthly_charges          █████████████         +0.25  (higher charges = more churn)
 tech_support_no          ███████████           +0.20  (no support = more churn)
 senior_citizen           ████████              +0.15  (seniors churn more)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

### Business Recommendations

1. **Target month-to-month customers** — Offer annual contract discounts (potential 15% churn reduction)
2. **Improve Fiber Optic support** — Fiber customers churn disproportionately; investigate service quality
3. **Proactive retention for tenure < 12 months** — New customers are highest risk
4. **Tech Support upsell** — Customers without tech support churn 2x more

---

## 🚀 Quick Start

### Installation

```bash
git clone https://github.com/datawithusman/telecom-churn-prediction.git
cd telecom-churn-prediction

pip install -r requirements.txt
```

### Run the Notebook

```bash
jupyter notebook churn_prediction.ipynb
```

### Or Run as Script

```python
from churn_model import ChurnPredictor

# Load and train
predictor = ChurnPredictor()
predictor.load_data("data/WA_Fn-UseC_-Telco-Customer-Churn.csv")
predictor.preprocess()
predictor.train()

# Evaluate
metrics = predictor.evaluate()
print(f"Accuracy: {metrics['accuracy']:.2%}")
print(f"F1 Score: {metrics['f1']:.2%}")

# Predict for a new customer
prediction = predictor.predict({
    "tenure": 2,
    "contract": "Month-to-month",
    "monthly_charges": 85.0,
    "internet_service": "Fiber optic",
    "tech_support": "No"
})
print(f"Churn Probability: {prediction['probability']:.1%}")
# Output: Churn Probability: 78.3%
```

---

## 📁 Project Structure

```
telecom-churn-prediction/
├── data/
│   └── WA_Fn-UseC_-Telco-Customer-Churn.csv
├── notebooks/
│   ├── 01_eda.ipynb              # Exploratory data analysis
│   ├── 02_preprocessing.ipynb    # Feature engineering
│   ├── 03_modeling.ipynb         # Model training & comparison
│   └── 04_insights.ipynb         # Business insights & visuals
├── src/
│   ├── __init__.py
│   ├── preprocessing.py          # Data cleaning pipeline
│   ├── models.py                 # Model definitions
│   └── visualization.py         # Plotting utilities
├── requirements.txt
└── README.md
```

---

## 📈 Exploratory Data Analysis Highlights

| Insight | Visualization |
|---------|--------------|
| Churn by contract type | Month-to-month customers churn **3x more** than annual |
| Tenure distribution | 50% of churn happens in first **10 months** |
| Monthly charges | Churned customers paid **$13 more** on average |
| Internet + Support | Fiber customers without tech support = **highest risk segment** |

---

## 👤 Author

**Muhammad Usman** — Data Scientist & AI Engineer

[![Portfolio](https://img.shields.io/badge/Portfolio-datawithusman.com-6C63FF?style=flat-square)](https://datawithusman.com)
[![Kaggle](https://img.shields.io/badge/Kaggle-20BEFF?style=flat-square&logo=kaggle&logoColor=white)](https://kaggle.com/datawithusman)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-0A66C2?style=flat-square&logo=linkedin&logoColor=white)](https://linkedin.com/in/datawithusman)
[![GitHub](https://img.shields.io/badge/GitHub-datawithusman-181717?style=flat-square&logo=github&logoColor=white)](https://github.com/datawithusman)

---

<div align="center">

**Found this analysis useful? Star ⭐ the repo and follow on [Kaggle](https://kaggle.com/datawithusman)!**

</div>