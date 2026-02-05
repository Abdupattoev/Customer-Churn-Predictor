import streamlit as st
import pandas as pd
import joblib

model = joblib.load("churn_model.pkl")

st.set_page_config(page_title="Customer Churn Predictor", layout="centered")

st.title("📊 Customer Churn Prediction App")
st.write("AI model that predicts if a telecom customer will churn.")

st.divider()

# ---- USER INPUTS ----
gender = st.selectbox("Gender", ["Female", "Male"])
senior = st.selectbox("Senior Citizen", ["No", "Yes"])
partner = st.selectbox("Has Partner?", ["No", "Yes"])
dependents = st.selectbox("Has Dependents?", ["No", "Yes"])
tenure = st.slider("Tenure (months)", 0, 72, 12)
monthly = st.number_input("Monthly Charges", 0.0, 200.0, 70.0)
total = st.number_input("Total Charges", 0.0, 10000.0, 1000.0)
contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])

# ---- ENCODING ----
gender = 1 if gender=="Male" else 0
senior = 1 if senior=="Yes" else 0
partner = 1 if partner=="Yes" else 0
dependents = 1 if dependents=="Yes" else 0

contract_map = {"Month-to-month":0,"One year":1,"Two year":2}
contract = contract_map[contract]

# ⚠️ MUST match model training columns exactly
input_dict = {
    "gender":gender,
    "SeniorCitizen":senior,
    "Partner":partner,
    "Dependents":dependents,
    "tenure":tenure,
    "MonthlyCharges":monthly,
    "TotalCharges":total,
    "Contract":contract
}

# fill rest columns automatically
for col in model.feature_names_in_:
    if col not in input_dict:
        input_dict[col] = 0

input_df = pd.DataFrame([input_dict])

# Create dataframe with ALL model features
full_input = {}

for col in model.feature_names_in_:
    if col in input_dict:
        full_input[col] = input_dict[col]
    else:
        full_input[col] = 0  # default for unused columns

input_df = pd.DataFrame([full_input])

# ---- Prediction ----
if st.button("Predict Churn"):
    try:
        pred = model.predict(input_df)[0]
        prob = model.predict_proba(input_df)[0][1]

        if pred == 1:
            st.error(f"⚠️ Customer likely to churn (probability {prob:.2f})")
        else:
            st.success(f"✅ Customer likely to stay (probability {prob:.2f})")

    except Exception as e:
        st.error(f"Model error: {e}")
import matplotlib.pyplot as plt

st.subheader("📊 Example: Churn Risk Visualization")

labels = ["Stay", "Churn"]
values = [1-prob if 'prob' in locals() else 0.5, prob if 'prob' in locals() else 0.5]

fig, ax = plt.subplots()
ax.bar(labels, values)
st.pyplot(fig)




