import streamlit as st
import joblib
import numpy as np
import pandas as pd
import plotly.express as px

# --- STEP 1: LOAD ASSETS ---
@st.cache_resource
def load_assets():
    # Aapki files ke names yahan match karne chahiye
    model = joblib.load("salary_model.pkl")
    job_enc = joblib.load("job_encoder.pkl")
    loc_enc = joblib.load("location_encoder.pkl")
    exp_enc = joblib.load("experience_encoder.pkl")
    df = pd.read_csv("Cleaned_Software_Engineer_Salaries-2.csv")
    return model, job_enc, loc_enc, exp_enc, df

model, job_encoder, loc_encoder, exp_encoder, df = load_assets()

# --- PAGE CONFIG ---
st.set_page_config(page_title="Salary Guide 2024", layout="wide")

# --- SIDEBAR: PREDICTION ---
st.sidebar.header("🔍 Predict Your Salary")
st.sidebar.write("Apni details select karein aur check karein market kya pay kar rahi hai.")

job_sel = st.sidebar.selectbox("Job Title", sorted(df['Job Title'].unique()))
loc_sel = st.sidebar.selectbox("Location", sorted(df['Location'].unique()))
exp_sel = st.sidebar.selectbox("Experience Level", sorted(df['Experience_Level'].unique()))

if st.sidebar.button("Check My Salary"):
    # Encoding inputs
    j_input = job_encoder.transform([job_sel])[0]
    l_input = loc_encoder.transform([loc_sel])[0]
    e_input = exp_encoder.transform([exp_sel])[0]
    
    # Predict
    features = np.array([[j_input, l_input, e_input]])
    prediction = model.predict(features)[0]
    
    st.sidebar.success(f"### 💰 Estimated Salary: ${int(prediction):,}")
    st.sidebar.info("Ye salary aapke select kiye hue features ke base par predict ki gayi hai.")

# --- MAIN DASHBOARD: EASY INSIGHTS ---
st.title("🚀 Software Engineer Salary Insights")
st.markdown("Is dashboard ke zariye aap dekh sakte hain ke industry mein salaries ka trend kya chal raha hai.")

# Top Metrics
m1, m2, m3 = st.columns(3)
m1.metric("Average Salary (Overall)", f"${int(df['Cleaned_Salary'].mean()):,}")
m2.metric("Most Frequent Job", df['Job Title'].mode()[0])
m3.metric("Highest Paying Level", "Senior Level")

st.divider()

# Row 1: Simple Comparisons
col1, col2 = st.columns(2)

with col1:
    st.subheader("📈 Experience ka Salary par Asar")
    st.write("Niche diye gaye graph se pata chalta hai ke experience barhne se salary mein kitna izafa hota hai.")
    avg_exp = df.groupby('Experience_Level')['Cleaned_Salary'].mean().sort_values().reset_index()
    fig1 = px.bar(avg_exp, x='Experience_Level', y='Cleaned_Salary', 
                 color='Experience_Level',
                 text_auto='.2s', # Bar ke upar numbers dikhayega
                 labels={'Cleaned_Salary': 'Average Salary ($)'})
    st.plotly_chart(fig1, use_container_width=True)

with col2:
    st.subheader("📍 Top 10 Cities for High Salaries")
    st.write("Ye wo shehar hain jahan software engineers ko sabse zyada pay kiya ja raha hai.")
    top_locs = df.groupby('Location')['Cleaned_Salary'].mean().nlargest(10).reset_index()
    fig2 = px.bar(top_locs, x='Cleaned_Salary', y='Location', 
                 orientation='h', 
                 color='Cleaned_Salary',
                 color_continuous_scale='Greens')
    st.plotly_chart(fig2, use_container_width=True)

# Row 2: Market Trend
st.divider()
st.subheader("📊 Salary Distribution")
st.write("The salary range of software engineers")
fig3 = px.histogram(df, x="Cleaned_Salary", nbins=20, 
                   labels={'Cleaned_Salary': 'Annual Salary Range'},
                   color_discrete_sequence=['#636EFA'])
st.plotly_chart(fig3, use_container_width=True)
