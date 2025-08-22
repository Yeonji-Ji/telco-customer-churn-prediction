# telco-customer-churn-prediction
## 1. Amazon Reviews Sentiment Analysis

### Goal

Build a machine learning model to classify Amazon product reviews as **positive or negative**.

This project demonstrates **text preprocessing, TF-IDF feature extraction, and baseline NLP models**.

### Business Context

Understanding customer sentiment enables companies to:

- Improve product feedback analysis
- Enhance customer service automation (chatbots, contact centers)
- Support data-driven product decisions

### Exploratory Data Analysis (EDA)

- **Label Distribution**: Balanced between positive and negative reviews
- **Text Length**: Positive reviews tend to be slightly longer
- **Most Frequent Words**: Clear separation between positive and negative vocabulary

### Models & Experiments

- **Logistic Regression**: Accuracy **0.89**, strong baseline
- **Naive Bayes**: Accuracy **0.86**, fast and interpretable
- **Linear SVM**: Accuracy **0.89**, strong discriminative power
- **XGBoost**: Accuracy ~**0.53**, struggled with sparse TF-IDF features

### Results & Insights

- **Linear SVM & Logistic Regression**: Best performing models
- **Naive Bayes**: Lightweight alternative
- **XGBoost**: Less effective on sparse, high-dimensional text data

---

## 2. Customer Churn Prediction

### Goal

Predict whether a customer will churn (leave the service) or remain.

This project demonstrates **feature engineering, handling class imbalance, and tree-based models**.

### Business Context

Customer churn directly impacts subscription-based businesses. Predicting churn helps:

- Identify at-risk customers
- Launch targeted retention campaigns
- Reduce long-term revenue loss

### Exploratory Data Analysis (EDA)

- **Churn Rate**: ~26% of customers churned (imbalanced data)
- **Tenure**: Shorter tenure → higher churn probability
- **Monthly Charges**: Higher monthly fees associated with churn
- **Total Charges**: Strongly correlated with tenure, distinct patterns by churn

### Models & Experiments

- **Logistic Regression (baseline)**
    - No balancing: Accuracy **0.81**, Recall **0.56** (missed many churned customers)
    - Balanced: Accuracy **0.74**, Recall **0.78** (better at capturing churn)
- **RandomForest**: Accuracy **0.78**, balanced Precision & Recall
- **XGBoost**: Accuracy **0.75**, slightly behind RandomForest

### Results & Insights

- **Logistic Regression (balanced)**: Best when Recall is critical (catching churn)
- **RandomForest**: Most balanced overall
- **XGBoost**: Comparable, with room for tuning

---

## Tech Stack

- Python (pandas, numpy, matplotlib, seaborn, scikit-learn, xgboost)
- Jupyter/Kaggle Notebook

---

## Next Steps

- Hyperparameter tuning for XGBoost & RandomForest
- Explore resampling methods (SMOTE, undersampling)
- Try ensemble models across projects
- Add deployment (Flask API / Streamlit dashboard)
