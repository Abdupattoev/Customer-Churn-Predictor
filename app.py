import streamlit as st
import pandas as pd
import joblib

# Load trained model
model = joblib.load("churn_model.pkl")

st.title("📊 Customer Churn Prediction App")

st.write("Enter customer details:")

# Create empty dataframe with SAME columns as model
feature_names = model.feature_names_in_
input_dict = {}

for feature in feature_names:
    input_dict[feature] = st.number_input(f"{feature}", value=0.0)

input_df = pd.DataFrame([input_dict])

if st.button("Predict"):
    prediction = model.predict(input_df)[0]
    prob = model.predict_proba(input_df)[0][1]

    if prediction == 1:
        st.error(f"Customer will churn 😢 (probability {prob:.2f})")
    else:
        st.success(f"Customer will stay 😎 (probability {prob:.2f})")


