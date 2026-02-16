import streamlit as st
import pandas as pd
import joblib

# ---- LOAD MODEL, SCALER, COLUMNS ----
model = joblib.load("final_model.pkl")
scaler = joblib.load("scaler.pkl")
columns = joblib.load("columns.pkl")

st.title("Customer Churn Prediction App")
st.write("Enter customer details to predict churn probability")

# ---- USER INPUT ----
gender = st.selectbox("Gender", ["Male", "Female"])
senior = st.selectbox("Senior Citizen", [0, 1])
partner = st.selectbox("Partner", ["Yes", "No"])
dependents = st.selectbox("Dependents", ["Yes", "No"])
tenure = st.number_input("Tenure Months", min_value=0, max_value=100, value=12)
phone_service = st.selectbox("Phone Service", ["Yes", "No"])
multiple_lines = st.selectbox("Multiple Lines", ["Yes", "No"])
internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
online_security = st.selectbox("Online Security", ["Yes", "No"])
online_backup = st.selectbox("Online Backup", ["Yes", "No"])
device_protection = st.selectbox("Device Protection", ["Yes", "No"])
tech_support = st.selectbox("Tech Support", ["Yes", "No"])
streaming_tv = st.selectbox("Streaming TV", ["Yes", "No"])
streaming_movies = st.selectbox("Streaming Movies", ["Yes", "No"])
contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
paperless_billing = st.selectbox("Paperless Billing", ["Yes", "No"])
payment_method = st.selectbox("Payment Method", ["Electronic check", "Mailed check", "Bank transfer", "Credit card"])
monthly_charges = st.number_input("Monthly Charges", min_value=0.0, value=70.0)
total_charges = st.number_input("Total Charges", min_value=0.0, value=1000.0)

# ---- CREATE DATAFRAME ----
data = pd.DataFrame([{
    "gender": gender,
    "Senior Citizen": senior,
    "Partner": partner,
    "Dependents": dependents,
    "Tenure Months": tenure,
    "Phone Service": phone_service,
    "Multiple Lines": multiple_lines,
    "Internet Service": internet_service,
    "Online Security": online_security,
    "Online Backup": online_backup,
    "Device Protection": device_protection,
    "Tech Support": tech_support,
    "Streaming TV": streaming_tv,
    "Streaming Movies": streaming_movies,
    "Contract": contract,
    "Paperless Billing": paperless_billing,
    "Payment Method": payment_method,
    "Monthly Charges": monthly_charges,
    "Total Charges": total_charges
}])

# ---- PREPROCESS DATA ----
data = pd.get_dummies(data, drop_first=True)
data = data.reindex(columns=columns, fill_value=0)
data_scaled = scaler.transform(data)

# ---- PREDICT ----
prediction = model.predict(data_scaled)
probability = model.predict_proba(data_scaled)

st.subheader("Prediction")
if prediction[0] == 0:
    st.success("Customer is unlikely to churn (0)")
else:
    st.error("Customer is likely to churn (1)")

st.subheader("Prediction Probability")
st.bar_chart(pd.DataFrame(probability, columns=["No Churn", "Churn"]))