# Telco Customer Churn Prediction


## Goal
Develop machine learning models to predict whether a telecom customer will churn based on service usage and demographic data.

---

## Data
[Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)
- Customer demographics: gender, age range, partner, dependents
- Customer account: tenure, contract type, payment method, billing, charges
- Services subscribed: phone, multiple lines, internet, online security, backup, device protection, tech support, streaming TV/movies
- Churn: whether the customer left in the last month

---

## Methods
- **EDA:** 
  - Churn Distribution: About 26% of customers left (imbalanced target).
  - Tenure: Customers with shorter tenure are more likely to churn.
  - Charges: Higher monthly charges increase churn probability.
  ![Churn Distribution](reports/figures/churn_vs_n.png)

- **Logistic Regression (baseline):**
  - Strong overall, but poor recall on minority churn class.  

- **Logistic Regression (balanced weights):**
  - Improved churn recall to 78%, but reduced precision.  

- **Logistic Regression + SMOTE**
  - Recall improved compared to baseline LogReg(70%), but precision still low.  

- **Random Forest**
  - Balanced performance (Precision 0.590, Recall 0.623).  

- **XGBoost**
  - Churn recall: 73% — stronger than Logistic Regression baseline.  

- **LightGBM**
  - Recall (70%) is similar to LogReg + SMOTE model.

- **Results Comparison**

| Model | Accuracy | Precision (Churn) | Recall (Churn) | F1 (Churn) |
| --- | --- | --- | --- | --- |
| Logistic Regression | 0.781 | 0.605 | 0.510 | 0.552 |
| LogReg (balanced) | 0.735 | 0.506 | 0.783 | 0.615 |
| LogReg + SMOTE | 0.723 | 0.484 | 0.700 | 0.572 |
| Random Forest | 0.785 | 0.590 | 0.623 | 0.610 |
| XGBoost | 0.756 | 0.529 | 0.725 | 0.612 |
| LightGBM | 0.759 | 0.534 | 0.700 | 0.605 |




**Business Context**  
Customer churn is a critical challenge for telecom providers. Identifying at-risk customers allows companies to take proactive actions (e.g., promotions, contract adjustments), reducing revenue loss and improving retention.


---

## Results Figures

### Confusion Matrices
- Logistic Regression  
  ![Logistic Regression Confusion Matrix](reports/figures/Baseline_Model_CM.png)  

- Random Forest  
  ![Random Forest Confusion Matrix](reports/figures/RandomForest.png/)  

- XGBoost  
  ![XGBoost Confusion Matrix](reports/figures/XGBoost.png)  

- LightGBM  
  ![LightGBM Confusion Matrix](reports/figures/LightGBM.png)  

---

## Key Takeaways

- **Balanced trade-off:** Random Forest achieves the best overall balance (Accuracy 0.785, F1 0.610), outperforming logistic regression variants in handling both precision and recall.

- **High recall options:** LogReg (balanced) and XGBoost yield higher recall (0.783 and 0.725), making them more suitable if capturing most churn cases is critical, though at the cost of precision and accuracy.

- **Modeling insight:** Simple Logistic Regression performs reasonably well, but tree-based methods (Random Forest, XGBoost, LightGBM) consistently provide stronger recall–precision balance for churn prediction.
