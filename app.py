import streamlit as st
import joblib
import numpy as np
import pandas as pd
import plotly.express as px

# --- STEP 1: LOAD RESOURCES ---
@st.cache_resource # Model load karne ke liye cache_resource behtar hai
def load_assets():
    model = joblib.load("salary_model.pkl")
    job_enc = joblib.load("job_encoder.pkl")
    loc_enc = joblib.load("location_encoder.pkl")
    exp_enc = joblib.load("experience_encoder.pkl")
    df = pd.read_csv("Cleaned_Software_Engineer_Salaries-2.csv")
    return model, job_enc, loc_enc, exp_enc, df

try:
    model, job_encoder, loc_encoder, exp_encoder, df = load_assets()
except Exception as e:
    st.error(f"Error loading files: {e}")

# --- PAGE SETUP ---
st.set_page_config(page_title="Salary Insights", layout="wide")

# Sidebar for Prediction
st.sidebar.header("🎯 Predict Your Salary")
job_input = st.sidebar.selectbox("Job Title", sorted(df['Job Title'].unique()))
loc_input = st.sidebar.selectbox("Location", sorted(df['Location'].unique()))
exp_input = st.sidebar.selectbox("Experience Level", sorted(df['Experience_Level'].unique()))

if st.sidebar.button("Predict Now"):
    # REAL PREDICTION LOGIC
    j_input = job_encoder.transform([job_input])[0]
    l_input = loc_encoder.transform([loc_input])[0]
    e_input = exp_encoder.transform([exp_input])[0]
    
    features = np.array([[j_input, l_input, e_input]])
    prediction = model.predict(features)[0]
    
    st.sidebar.balloons()
    st.sidebar.success(f"### Predicted Salary: ${int(prediction):,}")

# --- MAIN DASHBOARD (BETTER VISUALS) ---
st.title("📊 Software Engineering Market Insights")

# KPI Metrics
m1, m2, m3 = st.columns(3)
m1.metric("Average Market Salary", f"${int(df['Cleaned_Salary'].mean()):,}")
m2.metric("Top Paying Job", df.groupby('Job Title')['Cleaned_Salary'].mean().idxmax())
m3.metric("Total Data Points", len(df))

st.divider()

# Layout for Charts
col1, col2 = st.columns(2)

with col1:
    st.write("### 💰 Salary Range by Experience")
    # Box plot user ko batata hai ke min, max aur median kya hai
    fig1 = px.box(df, x="Experience_Level", y="Cleaned_Salary", 
                 color="Experience_Level", 
                 points="all", # Taake outliers bhi dikhein
                 title="Detailed Salary Spread")
    st.plotly_chart(fig1, use_container_width=True)

with col2:
    st.write("### 🏆 Top 10 Highest Paying Roles")
    top_jobs = df.groupby('Job Title')['Cleaned_Salary'].mean().sort_values(ascending=False).head(10).reset_index()
    fig2 = px.bar(top_jobs, x='Cleaned_Salary', y='Job Title', 
                 orientation='h', color='Cleaned_Salary',
                 color_continuous_scale='Viridis')
    st.plotly_chart(fig2, use_container_width=True)

st.write("### 📍 Location-wise Salary Analysis")
avg_loc = df.groupby('Location')['Cleaned_Salary'].mean().sort_values(ascending=False).head(15).reset_index()
fig3 = px.area(avg_loc, x="Location", y="Cleaned_Salary", 
              title="Average Salary in Top 15 Locations")
st.plotly_chart(fig3, use_container_width=True)
