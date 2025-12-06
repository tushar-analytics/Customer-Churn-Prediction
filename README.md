📦 E-Commerce Customer Churn Prediction

Predicting whether a customer will churn based on behavioral and demographic features.

🚀 Project Overview

Customer churn is a major challenge for e-commerce companies. This project builds a machine learning pipeline to predict whether a customer is likely to churn using multiple models and selects the best-performing one (XGBoost).

The project includes:

Data cleaning & preprocessing

Exploratory Data Analysis (EDA)

Feature engineering

Model training & hyperparameter tuning

SHAP explainability

Model deployment-ready prediction script

📁 Dataset

The dataset contains customer-level information such as:

Tenure

Complain

NumberOfAddress

CashbackAmount

MaritalStatus (encoded as 1 = Married, 2 = Single, 3 = Divorced)

WarehouseToHome

DaySinceLastOrder

SatisfactionScore

CityTier

Target variable: Churn

File used: E_Comm.csv

🧹 Data Preprocessing

Steps performed:

Handling missing values

Outlier clipping using np.clip()

Categorical encoding (LabelEncoder, Manual mapping)

Train-test split

Scaling using StandardScaler

Building ML pipelines

Example of marital status mapping:

marital_map = {"Married": 1, "Single": 2, "Divorced": 3}

📊 Exploratory Data Analysis

Several visualizations were created including:

Distribution plots

Correlation heatmaps

Churn ratio pie chart

Boxplots for numerical variables

ROC curve comparison of multiple models

Example for ROC subplot comparison:

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

RocCurveDisplay.from_estimator(log_reg_grid, X_test, y_test, ax=axes[0])
axes[0].set_title("Logistic Regression")

RocCurveDisplay.from_estimator(rfpipeline, X_test, y_test, ax=axes[1])
axes[1].set_title("Random Forest")

RocCurveDisplay.from_estimator(xgbpipe, X_test, y_test, ax=axes[2])
axes[2].set_title("XGBoost")

plt.tight_layout()
plt.show()

🤖 Model Building

Multiple models were tested:

Logistic Regression

Random Forest

XGBoost (best model)

🔧 Best XGBoost Hyperparameters
n_estimators = 2000  
learning_rate = 0.08  
max_depth = 6  
subsample = 0.8  
colsample_bytree = 0.8  
random_state = 42  
eval_metric = "logloss"


This configuration delivered the highest performance.

🏆 Best Model

The final model used is XGBoost inside a Scikit-Learn Pipeline.
Saved using:

joblib.dump(best_xgb_model, "xgb_churn_model.pkl")


best_xgb_model = final trained XGBoost pipeline.

🔍 Explainability (SHAP)

Since SHAP TreeExplainer does not support sklearn pipelines directly, model input was passed using:

Extracting underlying booster

Creating SHAP plots for feature impact

Feature importance was evaluated using both:

XGBoost built-in feature importance

SHAP summary plots

🔮 Prediction Function

A deployment-ready function was built to accept raw user input and convert it into the correct model format.

Example: Mapping user selections for Marital Status:

marital_map = {"Married": 1, "Single": 2, "Divorced": 3}
input_val = marital_map[user_input]

🗂️ Project Structure
├── E_Comm.csv                  # Dataset
├── churn_notebook.ipynb        # Full analysis & model building
├── xgb_churn_model.pkl         # Saved best model
├── predict.py                  # Prediction script (optional)
├── README.md                   # Project documentation

📈 Results

XGBoost achieved the best ROC-AUC score

Feature importance highlighted: Tenure, SatisfactionScore, CashbackAmount, CityTier

Model is ready for API/Streamlit deployment

🧪 How to Use
Install Required Libraries
pip install -r requirements.txt

Run Prediction
from joblib import load
model = load("xgb_churn_model.pkl")
prediction = model.predict(input_data)

💡 Future Improvements

Deploy with a Streamlit web app

Hyperparameter tuning using Optuna

Add SMOTE balancing

Build dashboard in Power BI

🤝 Contributing

Contributions, suggestions, and improvements are welcome!

⭐ Show Support

If you find this project useful, please ⭐ the repository!
