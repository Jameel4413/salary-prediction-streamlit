import streamlit as st
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Page Config
st.set_page_config(page_title="Salary Predictor", layout="wide") # Layout wide kar diya taake graphs ache dikhein

# --- STEP 1: LOAD DATA & MODELS ---
@st.cache_data # Taake baar baar load na ho
def load_resources():
    model = joblib.load("salary_model.pkl")
    job_encoder = joblib.load("job_encoder.pkl")
    loc_encoder = joblib.load("location_encoder.pkl")
    exp_encoder = joblib.load("experience_encoder.pkl")
    df = pd.read_csv("Cleaned_Software_Engineer_Salaries-2.csv")
    return model, job_encoder, loc_encoder, exp_encoder, df

model, job_encoder, loc_encoder, exp_encoder, df = load_resources()

# --- STEP 2: PREDICTION UI ---
st.title("💼 Software Engineer Salary Prediction App")

col1, col2 = st.columns([1, 1]) # Screen ko 2 hisso mein baanta

with col1:
    st.subheader("Enter Details")
    job = st.selectbox("Select Job Title", sorted(df['Job Title'].unique()))
    location = st.selectbox("Select Location", sorted(df['Location'].unique()))
    experience = st.selectbox("Select Experience Level", sorted(df['Experience_Level'].unique()))
    
    if st.button("🔮 Predict Salary"):
        job_enc = job_encoder.transform([job])[0]
        loc_enc = loc_encoder.transform([location])[0]
        exp_enc = exp_encoder.transform([experience])[0]
        features = np.array([[job_enc, loc_enc, exp_enc]])
        prediction = model.predict(features)[0]
        st.success(f"💰 Estimated Salary: ${int(prediction):,}")

# --- STEP 3: VISUALIZATIONS (EDA SECTION) ---
st.divider() # Ek line draw karega
st.header("📊 Data Insights & Visualizations")

tab1, tab2, tab3 = st.tabs(["Salary Distribution", "Top Trends", "Experience Analysis"])

with tab1:
    st.subheader("Salary Ki Distribution")
    fig1, ax1 = plt.subplots(figsize=(10, 5))
    sns.histplot(df['Cleaned_Salary'], kde=True, bins=25, ax=ax1, color='skyblue')
    st.pyplot(fig1) #plt.show() ki jagah ye use karein

with tab2:
    st.subheader("Top 10 Highest Paying Locations")
    top_salary_locations = df.groupby("Location")['Cleaned_Salary'].mean().sort_values(ascending=False).head(10)
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    sns.barplot(x=top_salary_locations.values, y=top_salary_locations.index, palette="mako", ax=ax2)
    st.pyplot(fig2)

with tab3:
    st.subheader("Salary Density by Experience Level")
    fig3, ax3 = plt.subplots(figsize=(10, 6))
    for level in df['Experience_Level'].unique():
        subset = df[df['Experience_Level'] == level]
        sns.kdeplot(subset['Cleaned_Salary'], label=level, fill=True, alpha=0.3, ax=ax3)
    plt.legend()
    st.pyplot(fig3)
