# -----------------------------
# Import Libraries
# -----------------------------
import streamlit as st
import numpy as np
import joblib
import plotly.graph_objects as go
import plotly.express as px

# -----------------------------
# Page Configuration
# -----------------------------
st.set_page_config(
    page_title="Heart Disease Prediction",
    page_icon="❤️",
    layout="wide"
)

# -----------------------------
# Load Model
# -----------------------------
@st.cache_resource
def load_model():
    model = joblib.load("heart_disease_model.pkl")
    scaler = joblib.load("heart_scaler.pkl")
    return model, scaler

model, scaler = load_model()

# -----------------------------
# Page Title
# -----------------------------
st.title("❤️ AI Heart Disease Prediction System")
st.markdown(
"""
This tool predicts the **probability of heart disease** based on patient medical data.
Enter patient details below and click **Predict**.
"""
)

st.markdown("---")

# -----------------------------
# Input Section
# -----------------------------
st.subheader("🧾 Patient Medical Information")

col1, col2, col3 = st.columns(3)

with col1:
    age = st.number_input("Age", 20, 100, 45)

    sex = st.selectbox(
        "Sex",
        ["Male", "Female"]
    )

    cp = st.selectbox(
        "Chest Pain Type",
        [
            "0 - Typical Angina",
            "1 - Atypical Angina",
            "2 - Non-anginal Pain",
            "3 - Asymptomatic"
        ]
    )

    trestbps = st.number_input(
        "Resting Blood Pressure",
        80, 200, 120
    )

    chol = st.number_input(
        "Cholesterol (mg/dl)",
        100, 600, 200
    )

with col2:
    fbs = st.selectbox(
        "Fasting Blood Sugar > 120 mg/dl",
        ["No", "Yes"]
    )

    restecg = st.selectbox(
        "Resting ECG Results",
        [
            "0 - Normal",
            "1 - ST-T abnormality",
            "2 - Left ventricular hypertrophy"
        ]
    )

    thalach = st.number_input(
        "Maximum Heart Rate Achieved",
        60, 220, 150
    )

    exang = st.selectbox(
        "Exercise Induced Angina",
        ["No", "Yes"]
    )

with col3:

    oldpeak = st.number_input(
        "ST Depression (Oldpeak)",
        0.0, 6.0, 1.0
    )

    slope = st.selectbox(
        "Slope of Peak Exercise ST Segment",
        [
            "0 - Upsloping",
            "1 - Flat",
            "2 - Downsloping"
        ]
    )

    ca = st.selectbox(
        "Number of Major Vessels",
        [0,1,2,3]
    )

    thal = st.selectbox(
        "Thalassemia",
        [
            "1 - Normal",
            "2 - Fixed Defect",
            "3 - Reversible Defect"
        ]
    )

st.markdown("---")

# -----------------------------
# Convert Inputs
# -----------------------------
sex = 1 if sex == "Male" else 0
fbs = 1 if fbs == "Yes" else 0
exang = 1 if exang == "Yes" else 0

cp = int(cp.split(" ")[0])
restecg = int(restecg.split(" ")[0])
slope = int(slope.split(" ")[0])
thal = int(thal.split(" ")[0])

# -----------------------------
# Prediction Button
# -----------------------------
predict = st.button("🔍 Predict Heart Disease Risk")

# -----------------------------
# Prediction Logic
# -----------------------------
if predict:

    input_data = np.array([[age,sex,cp,trestbps,chol,fbs,
                            restecg,thalach,exang,
                            oldpeak,slope,ca,thal]])

    input_scaled = scaler.transform(input_data)

    prediction = model.predict(input_scaled)
    probability = model.predict_proba(input_scaled)

    risk_percent = probability[0][1] * 100
    healthy_percent = probability[0][0] * 100

    st.markdown("---")
    st.header("📊 Prediction Results")

    colA, colB = st.columns([2,1])

# -----------------------------
# Result Message
# -----------------------------
    with colA:

        if risk_percent < 30:
            st.success("🟢 LOW RISK of Heart Disease")

        elif risk_percent < 70:
            st.warning("🟡 MODERATE RISK of Heart Disease")

        else:
            st.error("🔴 HIGH RISK of Heart Disease")

        st.write(f"### Heart Disease Probability: {risk_percent:.2f}%")

# -----------------------------
# Interactive Gauge Meter
# -----------------------------
    with colB:

        gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=risk_percent,
            number={'suffix': "%"},
            title={'text': "Risk Level"},
            gauge={
                'axis': {'range': [0,100]},
                'bar': {'color': "darkred"},
                'steps': [
                    {'range':[0,30],'color':"#2ecc71"},
                    {'range':[30,70],'color':"#f1c40f"},
                    {'range':[70,100],'color':"#e74c3c"}
                ]
            }
        ))

        st.plotly_chart(gauge, use_container_width=True)

# -----------------------------
# Probability Comparison Chart
# -----------------------------
    st.subheader("📈 Prediction Probability")

    prob_data = {
        "Category":["No Heart Disease","Heart Disease"],
        "Probability":[healthy_percent,risk_percent]
    }

    fig = px.bar(
        prob_data,
        x="Category",
        y="Probability",
        color="Category",
        color_discrete_sequence=["#2ecc71","#e74c3c"],
        text="Probability"
    )

    fig.update_traces(texttemplate='%{text:.2f}%', textposition='outside')

    fig.update_layout(
        yaxis_range=[0,100],
        title="Prediction Confidence",
        height=400
    )

    st.plotly_chart(fig, use_container_width=True)

# -----------------------------
# Medical Insight Section
# -----------------------------
    st.markdown("---")
    st.subheader("🧠 Medical Insight")

    insights = []

    if chol > 240:
        insights.append("High cholesterol increases heart disease risk.")

    if trestbps > 130:
        insights.append("High resting blood pressure detected.")

    if oldpeak > 2:
        insights.append("High ST depression indicates possible heart stress.")

    if thalach < 100:
        insights.append("Low maximum heart rate may indicate heart issues.")

    if len(insights) == 0:
        st.success("No major risk indicators detected in input data.")

    else:
        for i in insights:
            st.warning(i)