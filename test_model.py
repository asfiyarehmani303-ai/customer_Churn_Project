import joblib
import pandas as pd

# Load everything
model = joblib.load("final_model.pkl")
scaler = joblib.load("scaler.pkl")
columns = joblib.load("columns.pkl")

print("All files loaded successfully!")

# Example new customer data (RAW format like original dataset)
new_data = pd.DataFrame([{
    "gender": "Female",
    "Senior Citizen": 0,
    "Partner": "Yes",
    "Dependents": "No",
    "Tenure Months": 12,
    "Phone Service": "Yes",
    "Multiple Lines": "No",
    "Internet Service": "DSL",
    "Online Security": "No",
    "Online Backup": "Yes",
    "Device Protection": "No",
    "Tech Support": "No",
    "Streaming TV": "No",
    "Streaming Movies": "No",
    "Contract": "Month-to-month",
    "Paperless Billing": "Yes",
    "Payment Method": "Electronic check",
    "Monthly Charges": 70.35,
    "Total Charges": 845.50
}])

# Apply same preprocessing
new_data = pd.get_dummies(new_data, drop_first=True)

# Match training columns
new_data = new_data.reindex(columns=columns, fill_value=0)

# Scale
new_data_scaled = scaler.transform(new_data)

# Predict
prediction = model.predict(new_data_scaled)

print("Prediction:", prediction)