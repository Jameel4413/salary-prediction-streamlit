import streamlit as st
import joblib
import numpy as np
import pandas as pd

# Load model & encoders
model = joblib.load("salary_model.pkl")
job_encoder = joblib.load("job_encoder.pkl")
loc_encoder = joblib.load("location_encoder.pkl")
exp_encoder = joblib.load("experience_encoder.pkl")

# Load data (for dropdown values)
df = pd.read_csv("Cleaned_Software_Engineer_Salaries-2.csv")

st.set_page_config(page_title="Salary Predictor", layout="centered")

st.title("💼 Software Engineer Salary Prediction App")
st.write("Predict salary based on Job Title, Location & Experience Level")

# User inputs
job = st.selectbox("Select Job Title", sorted(df['Job Title'].unique()))
location = st.selectbox("Select Location", sorted(df['Location'].unique()))
experience = st.selectbox("Select Experience Level", sorted(df['Experience_Level'].unique()))

if st.button("🔮 Predict Salary"):
    # Encode inputs
    job_enc = job_encoder.transform([job])[0]
    loc_enc = loc_encoder.transform([location])[0]
    exp_enc = exp_encoder.transform([experience])[0]

    # Prepare input
    features = np.array([[job_enc, loc_enc, exp_enc]])

    # Prediction
    prediction = model.predict(features)[0]

    st.success(f"💰 Estimated Salary: ${int(prediction):,}")
