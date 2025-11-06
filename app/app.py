import streamlit as st
import numpy as np
import pandas as pd
import joblib

# Streamlit cache decorator to store resources (like models or database connections)
# so they load only once and stay in memory across app reruns.
# This helps improve performance by preventing repeated loading.
@st.cache_resource

# -----------------------------------------------------------
# 1. Load the saved model and scaler
# -----------------------------------------------------------
def load_resources():
    model = joblib.load("final_logistic_regression.pkl")
    scaler = joblib.load("scaler.pkl")
    return model, scaler

model, scaler = load_resources()

# -----------------------------------------------------------
# 2. Page configuration
# -----------------------------------------------------------
st.set_page_config(page_title="Heart Disease Risk Predictor", page_icon="❤️", layout="centered")

st.title("❤️ Heart Disease Risk Prediction App")
st.markdown("""
This app uses a **trained Machine Learning model** to predict the likelihood of heart disease 
based on basic, non-invasive clinical data.
""")

st.divider()

# -----------------------------------------------------------
# 3. Collect user input
# -----------------------------------------------------------
st.subheader("Enter Patient Information")

col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age (years)", min_value=20, max_value=100, value=50)
    gender = st.selectbox("Gender", ("Male", "Female"))
    chest_pain_type = st.selectbox("Chest Pain Type", 
                                   ("Typical Angina", "Atypical Angina", "Non-Anginal Pain", "Asymptomatic"))
    resting_blood_pressure = st.number_input("Resting Blood Pressure (mmHg)", min_value=80, max_value=200, value=120)
    cholesterol_measure = st.number_input("Serum Cholesterol (mg/dL)", min_value=100, max_value=600, value=240)
    fasting_blood_sugar = st.selectbox("Fasting Blood Sugar > 120 mg/dl", ("False", "True"))

with col2:
    resting_ecg_result = st.selectbox("Resting ECG Results", ("Normal", "ST-T Wave Abnormality", "Left Ventricular Hypertrophy"))
    max_heart_rate = st.number_input("Maximum Heart Rate Achieved (bpm)", min_value=60, max_value=210, value=150)
    exercise_induced_angina = st.selectbox("Exercise-Induced Angina", ("No", "Yes"))
    st_depression = st.number_input("ST Depression (oldpeak)", min_value=0.0, max_value=10.0, value=1.0)
    st_slope = st.selectbox("ST Slope", ("Upsloping", "Flat", "Downsloping"))
    major_vessels_count = st.slider("Number of Major Vessels (0–4)", 0, 4, 0)
    thal_status = st.selectbox("Thalassemia Status", ("Normal", "Fixed Defect", "Reversible Defect", "Unknown"))

# -----------------------------------------------------------
# 4. Encode categorical inputs
# -----------------------------------------------------------
gender = 1 if gender == "Male" else 0
fasting_blood_sugar = 1 if fasting_blood_sugar == "True" else 0
exercise_induced_angina = 1 if exercise_induced_angina == "Yes" else 0

chest_pain_map = {
    "Typical Angina": 0,
    "Atypical Angina": 1,
    "Non-Anginal Pain": 2,
    "Asymptomatic": 3
}
resting_ecg_map = {
    "Normal": 0,
    "ST-T Wave Abnormality": 1,
    "Left Ventricular Hypertrophy": 2
}
st_slope_map = {"Upsloping": 0, "Flat": 1, "Downsloping": 2}
thal_map = {"Normal": 0, "Fixed Defect": 1, "Reversible Defect": 2, "Unknown": 3}

chest_pain_type = chest_pain_map[chest_pain_type]
resting_ecg_result = resting_ecg_map[resting_ecg_result]
st_slope = st_slope_map[st_slope]
thal_status = thal_map[thal_status]

# -----------------------------------------------------------
# 5. Prepare data for prediction
# -----------------------------------------------------------
input_data = np.array([[age, gender, chest_pain_type, resting_blood_pressure, cholesterol_measure, fasting_blood_sugar,
                        resting_ecg_result, max_heart_rate, exercise_induced_angina, st_depression,
                        st_slope, major_vessels_count, thal_status]])

scaled_data = scaler.transform(input_data)

# -----------------------------------------------------------
# 6. Make prediction
# -----------------------------------------------------------
if st.button("Predict Heart Disease Risk"):
    prediction = model.predict(scaled_data)[0]
    probability = model.predict_proba(scaled_data)[0][1]

    st.divider()
    if prediction == 1:
        st.error(f"🚨 High Risk of Heart Disease ({probability*100:.1f}% probability)")
        st.markdown("**Recommendation:** Please consult a cardiologist for further examination.")
    else:
        st.success(f"✅ Low Risk of Heart Disease ({probability*100:.1f}% probability)")
        st.markdown("**Recommendation:** Maintain a healthy lifestyle and regular check-ups.")
