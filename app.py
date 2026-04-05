import streamlit as st
import numpy as np
import joblib

# ---------------- PAGE CONFIG ---------------- #
st.markdown(
    "<h1 style='text-align: center; font-size: 100px; color:#e63946;'>❤️ Heart Disease Prediction App</h1>",
    unsafe_allow_html=True
)
# ---------------- CUSTOM CSS ---------------- #
st.markdown("""
    <style>
    .big-title {
        font-size: 110px;   /* 🔥 Increased size */
        font-weight: 900;   /* Extra bold */
        color: #e63946;
        text-align: center;
        line-height: 1.1;
    }
    </style>
""", unsafe_allow_html=True)
# ---------------- LOAD MODEL ---------------- #
model = joblib.load("model/model.pkl")
scaler = joblib.load("model/scaler.pkl")


# ---------------- INPUT LAYOUT ---------------- #
col1, col2 = st.columns(2)

with col1:
    st.markdown('<div class="card">', unsafe_allow_html=True)

    age = st.number_input("Age", 1, 120, 30)

    sex_label = st.selectbox("Sex", ["Female", "Male"])
    sex = 0 if sex_label == "Female" else 1

    cp_options = {
        "Typical Angina": 0,
        "Atypical Angina": 1,
        "Non-anginal Pain": 2,
        "Asymptomatic": 3
    }
    cp_label = st.selectbox("Chest Pain Type", list(cp_options.keys()))
    cp = cp_options[cp_label]

    trestbps = st.number_input("Resting Blood Pressure", 80, 200, 120)
    chol = st.number_input("Cholesterol", 100, 400, 200)

    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    st.markdown('<div class="card">', unsafe_allow_html=True)

    fbs_label = st.selectbox("Fasting Blood Sugar > 120 mg/dl", ["No", "Yes"])
    fbs = 1 if fbs_label == "Yes" else 0

    restecg_options = {
        "Normal": 0,
        "ST-T wave abnormality": 1,
        "Left ventricular hypertrophy": 2
    }
    restecg_label = st.selectbox("Rest ECG", list(restecg_options.keys()))
    restecg = restecg_options[restecg_label]

    thalach = st.number_input("Max Heart Rate", 60, 220, 150)

    exang_label = st.selectbox("Exercise Induced Angina", ["No", "Yes"])
    exang = 1 if exang_label == "Yes" else 0

    oldpeak = st.number_input("ST Depression", 0.0, 10.0, 1.0)

    slope_options = {
        "Upsloping": 0,
        "Flat": 1,
        "Downsloping": 2
    }
    slope_label = st.selectbox("Slope", list(slope_options.keys()))
    slope = slope_options[slope_label]

    ca = st.selectbox("Major Vessels (0-3)", [0, 1, 2, 3])

    thal_options = {
        "Normal": 1,
        "Fixed Defect": 2,
        "Reversible Defect": 3
    }
    thal_label = st.selectbox("Thal", list(thal_options.keys()))
    thal = thal_options[thal_label]

    st.markdown('</div>', unsafe_allow_html=True)

# ---------------- BUTTON ---------------- #
st.markdown("<br>", unsafe_allow_html=True)
center_col = st.columns([1,2,1])

with center_col[1]:
    predict_btn = st.button("🔍 Predict", use_container_width=True)

# ---------------- PREDICTION ---------------- #
if predict_btn:
    input_data = np.array([[age, sex, cp, trestbps, chol, fbs,
                            restecg, thalach, exang, oldpeak,
                            slope, ca, thal]])

    input_data = scaler.transform(input_data)
    prediction = model.predict(input_data)

    st.markdown("<br>", unsafe_allow_html=True)

    if prediction[0] == 1:
        st.error("⚠️ High Risk of Heart Disease")
    else:
        st.success("✅ Low Risk of Heart Disease")