import streamlit as st
import pandas as pd
import joblib

# Load the model
model = joblib.load("xgb_churn_model.pkl")

st.set_page_config(page_title="Customer Churn Predictor", layout="wide")
st.title("📦 Customer Churn Prediction App")

st.write("Fill the customer details below to predict the probability of churn.")

# Mapping for Marital Status
marital_map = {
    "Married": 1,
    "Single": 2,
    "Divorced": 3
}

# --------------------------
# Input UI
# --------------------------
with st.form("churn_form"):
    col1, col2, col3 = st.columns(3)

    with col1:
        Tenure = st.number_input("Tenure (months)", min_value=0, max_value=120, step=1)
        Complain = st.selectbox("Customer Complained?", [0, 1])
        NumberOfAddress = st.number_input("Number of Addresses", min_value=1, max_value=20, step=1)

    with col2:
        CashbackAmount = st.number_input("Cashback Amount", min_value=0, max_value=5000, step=10)
        MaritalStatus_input = st.selectbox("Marital Status", ["Married", "Single", "Divorced"])
        WarehouseToHome = st.number_input("Distance Warehouse → Home (km)", min_value=0, max_value=50, step=1)

    with col3:
        DaySinceLastOrder = st.number_input("Days Since Last Order", min_value=0, max_value=365, step=1)
        SatisfactionScore = st.number_input("Satisfaction Score (1–5)", min_value=1, max_value=5, step=1)
        CityTier = st.selectbox("City Tier", [1, 2, 3])

    submitted = st.form_submit_button("Predict Churn")

# --------------------------
# Prediction Logic
# --------------------------
if submitted:

    MaritalStatus = marital_map[MaritalStatus_input]

    input_data = {
        "Tenure": Tenure,
        "Complain": Complain,
        "NumberOfAddress": NumberOfAddress,
        "CashbackAmount": CashbackAmount,
        "MaritalStatus": MaritalStatus,
        "WarehouseToHome": WarehouseToHome,
        "DaySinceLastOrder": DaySinceLastOrder,
        "SatisfactionScore": SatisfactionScore,
        "CityTier": CityTier
    }

    # Convert to DataFrame
    df = pd.DataFrame([input_data])

    # Predict
    pred = model.predict(df)[0]
    prob = model.predict_proba(df)[0][1]

    st.subheader("🔍 Prediction Result")

    if pred == 1:
        st.error(f"❌ Customer is likely to CHURN\n\n**Churn Probability: {prob:.2f}**")
    else:
        st.success(f"✅ Customer is NOT likely to churn\n\n**Churn Probability: {prob:.2f}**")
