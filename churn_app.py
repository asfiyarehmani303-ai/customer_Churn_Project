import pandas as pd
import joblib
import streamlit as st

# Load model and scaler
model = joblib.load("final_model.pkl")
scaler = joblib.load("scaler.pkl")
columns = joblib.load("columns.pkl")  # saved list of training columns

st.title("Customer Churn Prediction")

# Input fields
tenure = st.number_input("Tenure (months)", min_value=0)
monthly_charges = st.number_input("Monthly Charges")
contract_type = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
# add other inputs as needed...

# Create a DataFrame for a single input
input_dict = {
    "tenure": tenure,
    "MonthlyCharges": monthly_charges,
    "Contract_Month-to-month": 1 if contract_type=="Month-to-month" else 0,
    "Contract_One year": 1 if contract_type=="One year" else 0,
    "Contract_Two year": 1 if contract_type=="Two year" else 0,
    # include other categorical one-hot columns as 0/1
}

input_df = pd.DataFrame([input_dict])

# Align columns with training data
input_df = input_df.reindex(columns=columns, fill_value=0)

# Scale numeric features
input_scaled = scaler.transform(input_df)

# Predict
prediction = model.predict(input_scaled)
probability = model.predict_proba(input_scaled)[0][1]

st.write(f"Prediction: {'Likely to churn' if prediction[0]==1 else 'Unlikely to churn'}")
st.write(f"Churn Probability: {probability:.2f}")
