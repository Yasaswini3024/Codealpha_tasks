# app.py

import streamlit as st
import joblib
import numpy as np

# Load trained model
model = joblib.load("model.pkl")

st.title("Heart Disease Prediction")

st.write("Enter the patient's details below:")

age = st.number_input("Age", 20, 100, 50)
sex = st.selectbox("Sex (0 = Female, 1 = Male)", [0, 1])
cp = st.number_input("Chest Pain Type (0-3)", 0, 3, 0)
trestbps = st.number_input("Resting Blood Pressure", 80, 200, 120)
chol = st.number_input("Cholesterol", 100, 600, 200)
fbs = st.selectbox("Fasting Blood Sugar > 120 (0/1)", [0, 1])
restecg = st.number_input("Rest ECG (0-2)", 0, 2, 0)
thalach = st.number_input("Maximum Heart Rate", 60, 220, 150)
exang = st.selectbox("Exercise Induced Angina (0/1)", [0, 1])
oldpeak = st.number_input("Oldpeak", 0.0, 10.0, 1.0)
slope = st.number_input("Slope (0-2)", 0, 2, 1)
ca = st.number_input("Number of Major Vessels (0-4)", 0, 4, 0)
thal = st.number_input("Thal (0-3)", 0, 3, 2)

if st.button("Predict"):
    input_data = np.array([[age, sex, cp, trestbps, chol, fbs,
                            restecg, thalach, exang, oldpeak,
                            slope, ca, thal]])

    prediction = model.predict(input_data)

    if prediction[0] == 1:
        st.error("The patient is likely to have heart disease.")
    else:
        st.success("The patient is unlikely to have heart disease.")
