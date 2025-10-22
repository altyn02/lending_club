# app.py — LendingClub Frontend (feature-driven EDA)
# Runs NOW with your merged dataset; later plug in final feature list.

import os, io, tempfile, json, requests
import numpy as np
import pandas as pd
import streamlit as st
import altair as alt
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="Lending Club", layout="wide", page_icon="💳")
alt.data_transformers.disable_max_rows()
pd.set_option("display.max_columns", 100)
pd.set_option("display.float_format", "{:.2f}".format)

# -------- Helpers --------
def df_info_text(df: pd.DataFrame) -> str:
    buf = io.StringIO()
    df.info(buf=buf)
    return buf.getvalue()

@st.cache_data(show_spinner=False)
def download_to_tmp(url: str) -> str:
    # cache large GitHub Release asset locally
    ext = ".csv" if url.endswith(".csv") else (".parquet" if url.endswith(".parquet") else "")
    fd, path = tempfile.mkstemp(suffix=ext); os.close(fd)
    with requests.get(url, stream=True, timeout=180) as r:
        r.raise_for_status()
        total = int(r.headers.get("Content-Length", 0))
        done = 0; chunk = 1024 * 1024
        prog = st.progress(0, text="Downloading…")
        with open(path, "wb") as f:
            for part in r.iter_content(chunk_size=chunk):
                if part:
                    f.write(part); done += len(part)
                    if total:
                        prog.progress(min(done/total, 1.0),
                                      text=f"Downloading… {done//(1024*1024)} / {total//(1024*1024)} MB")
        prog.empty()
    return path

def load_df(upload, url: str) -> pd.DataFrame:
    if upload is not None:
        return pd.read_parquet(upload) if upload.name.endswith(".parquet") else pd.read_csv(upload, low_memory=False)
    local = download_to_tmp(url)
    return pd.read_parquet(local) if local.endswith(".parquet") else pd.read_csv(local, low_memory=False)

def to_float_pct(series: pd.Series) -> pd.Series:
    s = (series.astype(str).str.replace("%","",regex=False).str.replace(",","",regex=False)
         .str.strip().str.extract(r"([-+]?\d*\.?\d+)", expand=False))
    return pd.to_numeric(s, errors="coerce")

# -------- Sidebar: data + options --------
with st.sidebar:
    st.title("💳 Data")
    mode = st.radio("Source", ["From URL", "Upload"], horizontal=True)
    default_url = "https://github.com/altyn02/lending_club/releases/download/lending/accepted_merged.csv"
    url = st.text_input("Dataset URL", value=default_url) if mode == "From URL" else None
    upload = st.file_uploader("Upload CSV/Parquet", type=["csv","parquet"]) if mode == "Upload" else None

    st.markdown("---")
    st.subheader("Sampling")
    sample_n = st.slider("Rows to analyze (for speed)", 50_000, 500_000, 200_000, 50_000)
    stratify = st.checkbox("Stratify by target (if set)", value=True)

    st.markdown("---")
    st.subheader("DTI cleaning (optional)")
    dti_fix = st.checkbox("Clean DTI range", value=True)
    dti_min, dti_max = 0, 99
    high_dti_flag = st.checkbox("Add high_dti_flag (>70%)", value=True)

# -------- Load data --------
if mode == "From URL" and not url:
    st.info("Paste a dataset URL or switch to Upload.")
    st.stop()
if mode == "Upload" and upload is None:
    st.info("Upload CSV or Parquet, or switch to URL.")
    st.stop()

try:
    df_full = load_df(upload, url)  # big
except Exception as e:
    st.error(f"Could not load dataset: {e}")
    st.stop()

st.success(f"Loaded {len(df_full):,} rows × {df_full.shape[1]} columns")

# -------- Target + feature controls --------
st.markdown("### 1) Target & Features")
cols = list(df_full.columns)
target_guess = "loan_status" if "loan_status" in cols else None
c1, c2 = st.columns([1,2])
with c1:
    target_col = st.selectbox("Target (optional)", [None] + cols, index=(cols.index(target_guess)+1 if target_guess in cols else 0))
with c2:
    # Preselect a reasonable set; you can change later
    preselect = [c for c in ["loan_amnt","int_rate","dti","annual_inc","grade","purpose","term","revol_util","fico_range_low","fico_range_high"] if c in cols]
    chosen = st.multiselect("Choose features to show", options=cols, default=preselect)

# -------- Light cleaning for types (safe) --------
for c in ["int_rate","revol_util","dti"]:
    if c in df_full.columns and not pd.api.types.is_numeric_dtype(df_full[c]):
        df_full[c] = to_float_pct(df_full[c])
if "issue_d" in df_full.columns and not np.issubdtype(df_full["issue_d"].dtype, np.datetime64):
    # try LC format first, then generic
    dt = pd.to_datetime(df_full["issue_d"], errors="coerce", format="%b-%Y")
    if dt.isna().all():
        dt = pd.to_datetime(df_full["issue_d"], errors="coerce")
    df_full["issue_d"] = dt

# Optional DTI cleaning/flag
if dti_fix and "dti" in df_full.columns:
    df_full.loc[(df_full["dti"] < dti_min) | (df_full["dti"] > dti_max), "dti"] = np.nan
if high_dti_flag and "dti" in df_full.columns:
    df_full["high_dti_flag"] = (df_full["dti"] > 70).astype("Int64")

# -------- Sampling (fast EDA) --------
if target_col and stratify and target_col in df_full.columns:
    # stratified sample (approx)
    df = (df_full.groupby(target_col, g
