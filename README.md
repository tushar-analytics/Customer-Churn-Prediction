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

---

## 🤖 **Model Building**

| **Model**              | **Status**             |
|------------------------|------------------------|
| Logistic Regression    | Baseline model         |
| Random Forest          | Good performance       |
| **XGBoost**            | **Best-performing model** |

---

## 🏆 **Best Model: XGBoost**

| **Hyperparameter** | **Value** |
|--------------------|-----------|
| n_estimators       | 2000      |
| learning_rate      | 0.08      |
| max_depth          | 6         |
| subsample          | 0.8       |
| colsample_bytree   | 0.8       |
| eval_metric        | logloss   |

### **Saving the Model**

```python
joblib.dump(best_xgb_model, "xgb_churn_model.pkl")
```

---

## 🔍 **Explainability (SHAP)**

| **Method**            | **Purpose**                               |
|-----------------------|---------------------------------------------|
| SHAP Summary Plot     | Reveals global feature importance          |
| Booster Extract       | Needed because TreeExplainer doesn't support pipelines |
| Feature Importance    | Confirms top predictors                    |

### **Top Predictive Features**
- Tenure  
- SatisfactionScore  
- CashbackAmount  
- CityTier  

---

## 🔮 **Prediction Pipeline**

✔ Accepts raw user input  
✔ Converts categorical values  
✔ Passes cleaned data into the model pipeline  

### **Example: Mapping User Input**

```python
input_val = marital_map[user_choice]
```

---

## 🗂️ **Project Structure**


| **File**               | **Description**            |
|------------------------|-----------------------------|
| E_Comm.csv             | Dataset                    |
| churn_notebook.ipynb   | Full workflow notebook     |
| xgb_churn_model.pkl    | Final trained model        |
| predict.py             | Input-to-prediction script |
| README.md              | Project documentation      |

---

## 📈 **Results**

| **Metric**  | **Observation**                |
|-------------|--------------------------------|
| ROC-AUC     | Highest for XGBoost            |
| Key Drivers | Tenure, SatisfactionScore, CashbackAmount |
| Deployment  | Model ready for API / Streamlit |

---

## **Dashboard**

<p align="center">
  <img src="Video Project 4.gif" width="700"/>
</p>

## 🧪 **How to Run**

### **Install Dependencies**
```bash
pip install -r requirements.txt
```

---

## 💡 **Future Enhancements**

| **Improvement**     | **Benefit**               |
|----------------------|---------------------------|
| Streamlit App       | User-friendly UI          |
| SMOTE               | Handle class imbalance    |
| Optuna              | Faster & better tuning    |
| Power BI Dashboard  | Business insights         |

---

## 🤝 **Contributing**

Feel free to submit issues or pull requests.

---

## ⭐ **Support**

If you found this project helpful, please ⭐ the repository!





