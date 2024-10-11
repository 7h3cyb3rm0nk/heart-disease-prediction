import streamlit as st
import numpy as np
import joblib
# from streamlit_lottie import st_lottie
import requests

knn = joblib.load("model_KNeighborsClassifier.joblib")
scaler = joblib.load("scaler.joblib")


def load_lottieurl(url: str):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()


st.set_page_config(
    page_title="Cardiovascular Risk Prediction", page_icon="❤️", layout="wide"
)

# Custom CSS with !important to override Streamlit's default styles
st.markdown(
    """
    <style>
    .stApp {
        background-color: #2b3035 !important;
    }
    .main {
        background-color: #2b3035 !important;
    }
    .stTextInput > div > div > input {
        background-color: #3d4449 !important;
        color: #ffffff !important;
    }
    .stSelectbox > div > div > div {
        background-color: #3d4449 !important;
        color: #ffffff !important;
    }
    .stNumberInput > div > div > input {
        background-color: #3d4449 !important;
        color: #ffffff !important;
    }
    .stButton > button {
        background-color: #ff4b4b !important;
        color: #ffffff !important;
    }
    .css-1adrfps {
        background-color: #2b3035 !important;
    }
    h1, h2, h3, p, label {
        color: #ffffff !important;
    }
    .stMarkdown {
        color: #ffffff !important;
    }
    .stProgress > div > div > div > div {
        background-color: #ff4b4b !important;
    }
    .stMetric {
        background-color: #3d4449 !important;
        color: #ffffff !important;
    }
    .stMetric > div {
        background-color: #3d4449 !important;
        color: #ffffff !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Lottie Animation
lottie_health = load_lottieurl(
    "https://assets3.lottiefiles.com/packages/lf20_5njp3vgg.json"
)

col1, col2 = st.columns([2, 1])

with col1:
    st.title("❤️ Cardiovascular Risk Prediction")
    st.write(
        "Enter your health information to predict your 10-year cardiovascular disease risk."
    )

# with col2:
#     st_lottie(lottie_health, height=200)

left_column, right_column = st.columns(2)

with left_column:
    age = st.number_input("Age", min_value=18, max_value=100, value=36)
    sex = st.selectbox("Sex", ["Male", "Female"])
    is_smoking = st.selectbox("Is Smoking", ["No", "Yes"])
    cigsPerDay = st.number_input(
        "Cigarettes per Day", min_value=0, max_value=100, value=0
    )
    BPMeds = st.selectbox("Blood Pressure Medication", ["No", "Yes"])
    prevalentStroke = st.selectbox("Prevalent Stroke", ["No", "Yes"])
    prevalentHyp = st.selectbox("Prevalent Hypertension", ["No", "Yes"])
    diabetes = st.selectbox("Diabetes", ["No", "Yes"])

with right_column:
    totChol = st.number_input(
        "Total Cholesterol", min_value=100, max_value=500, value=212
    )
    BMI = st.number_input("BMI", min_value=10, max_value=50, value=20)
    heartRate = st.number_input("Heart Rate", min_value=40, max_value=150, value=75)
    glucose = st.number_input("Glucose", min_value=50, max_value=300, value=75)
    education = st.selectbox(
        "Education Level",
        ["Below 10th", "10th/SSLC", "12th Standard/HSC", "Graduate/Post Graduate"],
    )
    pulse_pressure = st.number_input(
        "Pulse Pressure", min_value=20, max_value=100, value=50
    )

# Convert categorical inputs to numerical
sex = 1 if sex == "Male" else 0
BPMeds = 1 if BPMeds == "Yes" else 0
prevalentStroke = 1 if prevalentStroke == "Yes" else 0
prevalentHyp = 1 if prevalentHyp == "Yes" else 0
diabetes = 1 if diabetes == "Yes" else 0
is_smoking = 1 if is_smoking == "Yes" else 0

education_dict = {
    "Below 10th": 1,
    "10th/SSLC": 2,
    "12th Standard/HSC": 3,
    "Graduate/Post Graduate": 4,
}

new_data = np.array(
    [
        [
            age,
            education_dict[education],
            sex,
            is_smoking,
            cigsPerDay,
            BPMeds,
            prevalentStroke,
            prevalentHyp,
            diabetes,
            totChol,
            BMI,
            heartRate,
            glucose,
            pulse_pressure,
        ]
    ],
    # dtype=np.float16,
)


if st.button("Predict Risk"):
    new_data_scaled = scaler.transform(new_data)

    prediction = knn.predict_proba(new_data_scaled)

    st.subheader("Prediction Result:" + str(prediction.shape) + str(prediction))
    risk_probability = prediction[:, 1][0]

    col1, col2 = st.columns(2)

    with col1:
        st.metric("Probability of 10-year CHD", f"{risk_probability:.2%}")

    with col2:
        risk_level = (
            "High"
            if risk_probability > 0.2
            else "Moderate" if risk_probability > 0.1 else "Low"
        )
        st.metric("Risk Assessment", risk_level)
    st.progress(risk_probability)

    st.subheader("Recommendations:")
    if risk_probability > 0.2:
        st.warning(
            "Your risk is considered high. Please consult with a healthcare professional for a thorough evaluation and personalized advice."
        )
    elif risk_probability > 0.1:
        st.warning("Your risk is considered moderate. Please take care of your health.")
    else:
        st.success(
            "Your risk is considered low. Continue maintaining a healthy lifestyle with regular exercise and a balanced diet."
        )

# Footer
st.markdown("---")
st.warning(
    "Note: This tool provides an estimate and should not replace professional medical advice."
)
