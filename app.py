
import streamlit as st
import numpy as np
import pandas as pd


import joblib

knn  = joblib.load('knnModel.joblib')
scaler = joblib.load('scaler.joblib')





# Streamlit UI
st.title("Cardiovascular Risk Prediction")

# Input fields for user data
age = st.number_input("Age", min_value=18, max_value=100, value=36)
sex = st.selectbox("Sex", ["Male", "Female"])
cigsPerDay = st.number_input("Cigarettes per Day", min_value=0, max_value=100, value=0)
BPMeds = st.selectbox("Blood Pressure Medication", ["No", "Yes"])
prevalentStroke = st.selectbox("Prevalent Stroke", ["No", "Yes"])
prevalentHyp = st.selectbox("Prevalent Hypertension", ["No", "Yes"])
diabetes = st.selectbox("Diabetes", ["No", "Yes"])
totChol = st.number_input("Total Cholesterol", min_value=100, max_value=500, value=212)
BMI = st.number_input("BMI", min_value=10, max_value=50, value=20)
heartRate = st.number_input("Heart Rate", min_value=40, max_value=150, value=75)
glucose = st.number_input("Glucose", min_value=50, max_value=300, value=75)
education = st.selectbox("Education Level", ["Below 10th", "10th/SSLC", "12th Standard/HSC", "Graduate/Post Graduate"])
pulse_pressure = st.number_input("Pulse Pressure", min_value=20, max_value=100, value=50)

# Convert categorical inputs to numerical
sex = 1 if sex == "Male" else 0
BPMeds = 1 if BPMeds == "Yes" else 0
prevalentStroke = 1 if prevalentStroke == "Yes" else 0
prevalentHyp = 1 if prevalentHyp == "Yes" else 0
diabetes = 1 if diabetes == "Yes" else 0

# Create a dictionary for education levels
education_dict = {"Below 10th": [1, 0, 0, 0], "10th/SSLC": [1, 1, 0, 0], "12th Standard/HSC": [1, 1, 1, 0], "Graduate/Post Graduate": [1, 1, 1, 1]}

# Create the new data dictionary
new_data = {'age': age,
            'sex': sex,
            'cigsPerDay': cigsPerDay,
            'BPMeds': BPMeds,
            'prevalentStroke': prevalentStroke,
            'prevalentHyp': prevalentHyp,
            'diabetes': diabetes,
            'totChol': totChol,
            'BMI': BMI,
            'heartRate': heartRate,
            'glucose': glucose,
            'education_1.0': education_dict[education][0],
            'education_2.0': education_dict[education][1],
            'education_3.0': education_dict[education][2],
            'education_4.0': education_dict[education][3],
            'pulse_pressure': pulse_pressure}

new_data = pd.DataFrame([new_data])

# Data Scaling
new_data_scaled = scaler.transform(new_data)

# Make the prediction
prediction = knn.predict_proba(new_data_scaled)

# Display the prediction
st.subheader("Prediction:")
st.write("Probability of developing 10-year CHD:", prediction[:, 1][0])
st.write("Risk Assessment:", "High Risk" if prediction[:, 1][0] > 0.5 else "Low Risk") 
