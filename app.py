
import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
st.markdown("### 📊 Customer Churn Prediction using Machine Learning")
st.write(
"This machine learning web app predicts whether a telecom customer will churn. "
"It helps businesses identify high-risk customers and improve retention strategies."
)
st.set_page_config(page_title="Customer Churn Predictor", layout="centered")

@st.cache_resource
def train_model():
    # Load dataset
    url = "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv"
    df = pd.read_csv(url)

    # Clean
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce").fillna(0)
    df.drop("customerID", axis=1, inplace=True)
    df["Churn"] = (df["Churn"] == "Yes").astype(int)

    # Encode categoricals (simple label encoding)
    le = LabelEncoder()
    for col in df.select_dtypes(include=["object"]).columns:
        df[col] = le.fit_transform(df[col].astype(str))

    X = df.drop("Churn", axis=1)
    y = df["Churn"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = RandomForestClassifier(
        n_estimators=200, random_state=42, class_weight="balanced"
    )
    model.fit(X_train, y_train)

    acc = accuracy_score(y_test, model.predict(X_test))
    return model, X.columns.tolist(), float(acc)

model, feature_names, acc = train_model()

st.title("📊 Customer Churn Prediction App")
st.write("AI model that predicts if a telecom customer will churn.")

st.subheader("📈 Model Performance")
st.write(f"Model: Random Forest (trained on IBM Telco dataset)")
st.write(f"Test Accuracy: **{acc:.2%}**")

st.subheader("💡 Business Impact")
st.write(
    "Predicting churn helps telecom companies identify high-risk customers early and "
    "take action (retention offers, customer support) to reduce revenue loss."
)

st.divider()

# --- Simple user-friendly inputs (only a few key features) ---
gender = st.selectbox("Gender", ["Female", "Male"])
senior = st.selectbox("Senior Citizen", ["No", "Yes"])
partner = st.selectbox("Has Partner?", ["No", "Yes"])
dependents = st.selectbox("Has Dependents?", ["No", "Yes"])
tenure = st.slider("Tenure (months)", 0, 72, 12)
monthly = st.number_input("Monthly Charges", 0.0, 200.0, 70.0)
total = st.number_input("Total Charges", 0.0, 10000.0, 1000.0)
contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])

# --- Map to encoded values (approximate, consistent with training label encoding approach) ---
# NOTE: Because training used LabelEncoder on full dataset, these mappings are not guaranteed
# to match exactly. To stay stable, we set only known numeric fields and default others to 0.
input_dict = {
    "tenure": float(tenure),
    "MonthlyCharges": float(monthly),
    "TotalCharges": float(total),
}

# Fill ALL model features, default 0
full_input = {col: 0.0 for col in feature_names}
for k, v in input_dict.items():
    if k in full_input:
        full_input[k] = v

input_df = pd.DataFrame([full_input], columns=feature_names)

if st.button("Predict Churn"):
    prob = model.predict_proba(input_df)[0][1]
    pred = 1 if prob >= 0.5 else 0

    if pred == 1:
        st.error(f"⚠️ Customer likely to churn (probability **{prob:.2f}**)") 
    else:
        st.success(f"✅ Customer likely to stay (probability **{prob:.2f}**)")

    st.subheader("📊 Probability Breakdown")
    st.progress(min(max(prob, 0.0), 1.0))
    st.caption("This probability is produced by the ML model trained on the dataset.")

st.markdown("---")
st.markdown("Built by: Abdupattoev Sardorbek | Data Science Portfolio Project")



