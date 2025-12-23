import streamlit as st
import joblib
import pandas as pd
import plotly.express as px # Interactive graphs ke liye

# --- Resources Load Karein ---
@st.cache_data
def load_data():
    df = pd.read_csv("Cleaned_Software_Engineer_Salaries-2.csv")
    model = joblib.load("salary_model.pkl")
    return df, model

df, model = load_data()

# Page Layout
st.set_page_config(page_title="Salary Insights", layout="wide")
st.title("📊 Software Engineer Salary Dashboard")

# --- SECTION 1: KEY METRICS ---
# Ye user ko foran bata dega ke overall market ka haal kya hai
st.subheader("Market at a Glance")
col_m1, col_m2, col_m3 = st.columns(3)
col_m1.metric("Average Salary", f"${int(df['Cleaned_Salary'].mean()):,}")
col_m2.metric("Highest Salary", f"${int(df['Cleaned_Salary'].max()):,}")
col_m3.metric("Top Location", df['Location'].mode()[0])

st.divider()

# --- SECTION 2: PREDICTION TOOL (Sidebar mein kar dete hain taake screen saaf rahe) ---
st.sidebar.header("Salary Predictor")
job = st.sidebar.selectbox("Job Title", sorted(df['Job Title'].unique()))
loc = st.sidebar.selectbox("Location", sorted(df['Location'].unique()))
exp = st.sidebar.selectbox("Experience", sorted(df['Experience_Level'].unique()))

if st.sidebar.button("Predict"):
    # (Yahan aapka prediction logic aayega jaisa pehle tha)
    st.sidebar.success("Predicted: $120k") # Example

# --- SECTION 3: USEFUL VISUALS ---
st.subheader("User-Friendly Insights")

c1, c2 = st.columns(2)

with c1:
    # Insight: Experience ke saath salary kaise barhti hai?
    st.write("### 📈 Experience vs Salary")
    fig_exp = px.box(df, x='Experience_Level', y='Cleaned_Salary', 
                     color='Experience_Level',
                     title="Salary Range by Experience (Hover to see details)",
                     labels={'Cleaned_Salary': 'Salary ($)', 'Experience_Level': 'Level'})
    st.plotly_chart(fig_exp, use_container_width=True)

with c2:
    # Insight: Top 10 cities jo sabse zyada pay kar rahi hain
    st.write("### 📍 Top 10 High-Paying Locations")
    top_10_loc = df.groupby('Location')['Cleaned_Salary'].mean().nlargest(10).reset_index()
    fig_loc = px.bar(top_10_loc, x='Cleaned_Salary', y='Location', 
                     orientation='h', color='Cleaned_Salary',
                     title="Where can you earn the most?")
    st.plotly_chart(fig_loc, use_container_width=True)

st.write("### 🔍 Salary Density")
# User dekh sakta hai ke zyada tar log kis bracket mein aate hain
fig_hist = px.histogram(df, x="Cleaned_Salary", nbins=30, 
                        title="Common Salary Brackets",
                        labels={'Cleaned_Salary': 'Annual Salary'},
                        color_discrete_sequence=['indianred'])
st.plotly_chart(fig_hist, use_container_width=True)
