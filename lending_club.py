import pandas as pd
import streamlit as st

st.set_page_config(page_title="Lending Club Dashboard")
st.title("💳 Lending Club Data Dashboard")

url = "https://github.com/altyn02/lending_club/releases/download/lending/accepted_merged.csv"

@st.cache_data
def load_data():
    return pd.read_csv(url)

with st.spinner("📥 Loading large CSV file..."):
    df = load_data()

st.success(f"✅ Loaded data with {df.shape[0]} rows and {df.shape[1]} columns.")
st.dataframe(df.head())
