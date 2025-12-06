# 📦 **E-Commerce Customer Churn Prediction**

Predicting customer churn for an e-commerce platform using machine learning (XGBoost, Random Forest, Logistic Regression).

---

## 📘 **Project Overview**

This project builds an end-to-end machine learning workflow for churn prediction:

- Data preprocessing  
- Exploratory Data Analysis (EDA)  
- Feature engineering  
- Model training and hyperparameter tuning  
- SHAP explainability  
- Deployment-ready prediction pipeline  

---

## 📁 **Dataset Information**

| **Feature Name**          | **Description** |
|---------------------------|-----------------|
| Tenure                    | Customer tenure in months |
| Complain                  | Complaint raised (0/1) |
| NumberOfAddress           | Total addresses used |
| CashbackAmount            | Cashback received |
| MaritalStatus             | 1 = Married, 2 = Single, 3 = Divorced |
| WarehouseToHome           | Distance from warehouse |
| DaySinceLastOrder         | Days since last order |
| SatisfactionScore         | Rating 1–5 |
| CityTier                  | Tier 1/2/3 city |
| Churn (Target)            | 0 = Not churned, 1 = Churned |

_File used_: **E_Comm.csv**

---

## 🧹 **Data Preprocessing**

| **Step** | **Description** |
|----------|-----------------|
| Missing Values | Handled using imputation/removal |
| Outliers | Clipped with `np.clip()` |
| Encoding | Custom mapping → Marital Status |
| Scaling | StandardScaler inside pipeline |
| Split | Train-test split (80/20) |

**Marital Status Mapping**

```python
marital_map = {"Married": 1, "Single": 2, "Divorced": 3}
```

---

## 📊 **Exploratory Data Analysis (EDA)**

| **Analysis Type**        | **What It Shows**            |
|--------------------------|-------------------------------|
| Distribution plots       | Spread and skewness          |
| Correlation heatmap      | Relationships between features |
| Churn ratio chart        | Class imbalance              |
| Boxplots                 | Outlier detection            |
| ROC curves               | Model comparison             |

---

### **ROC Curve Subplot Example**

```python
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

RocCurveDisplay.from_estimator(log_reg_grid, X_test, y_test, ax=axes[0])
axes[0].set_title("Logistic Regression")

RocCurveDisplay.from_estimator(rfpipeline, X_test, y_test, ax=axes[1])
axes[1].set_title("Random Forest")

RocCurveDisplay.from_estimator(xgbpipe, X_test, y_test, ax=axes[2])
axes[2].set_title("XGBoost")

plt.tight_layout()
plt.show()
```



