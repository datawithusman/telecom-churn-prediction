Telecom Customer Churn Prediction
📌 Project Overview

This project focuses on predicting customer churn in a telecom company using Machine Learning. The goal is to identify customers who are likely to leave the service so that the company can take preventive retention measures.

📊 Dataset Information

Total Records: 7043 customers

Features: Demographics, billing information, services subscribed

Target Variable: Churn Value (0 = No churn, 1 = Churn)

⚙️ Challenges Faced
1️⃣ Incorrect Data Type Handling

The "Total Charges" column was incorrectly stored as an object type, which caused improper one-hot encoding and generated thousands of unnecessary features.

2️⃣ Class Imbalance

Churn customers were significantly fewer than non-churn customers, causing the model to bias toward majority class predictions.

3️⃣ Accuracy vs Recall Trade-off

Initial model had 80% accuracy but only 58% churn recall, which was not acceptable from a business perspective.

🛠️ Approach & Methodology

Data cleaning and preprocessing

Converted "Total Charges" to numeric

Handled missing values

One-hot encoding for categorical variables

Feature scaling using StandardScaler

Train-test split (80-20)

Logistic Regression model

Applied class_weight='balanced' to handle imbalance

Evaluated using Confusion Matrix, Classification Report, ROC Curve

📈 Model Performance
Metric	Value
Accuracy	75%
Churn Recall	81%
AUC Score	0.85

The final model successfully identifies 81% of customers who are likely to churn.

💡 Key Business Insights

Higher Total Charges increase churn risk

Fiber Optic internet users show higher churn probability

Customers using Electronic Check payment method are more likely to churn

Paperless billing customers have higher churn tendency

🚀 Business Impact

This model can help telecom companies proactively target high-risk customers with retention offers, reducing revenue loss and improving customer lifetime value.

🧠 Technologies Used

Python

Pandas

NumPy

Scikit-learn

Matplotlib

📂 Repository Contents

Jupyter Notebook (.ipynb)

Trained model file (churn_model.pkl)

README documentation

👤 Author

Muhammad Usman
