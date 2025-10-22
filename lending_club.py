# app.py — Lending Club Design-First Dashboard
# minimal, aesthetic EDA with your merged dataset (URL or upload)

import os, io, tempfile, requests
import numpy as np
import pandas as pd
import streamlit as st
import altair as alt
from datetime import datetime

# -------------------- Page & Theme --------------------
st.set_page_config(page_title="Lending Club 💳", page_icon="💳", layout="wide")
alt.data_transformers.disable_max_rows()
pd.set_option("display.max_columns", 200)

# -------------------- CSS (Design) --------------------
CSS = """
<style>
/* global */
html, body, [class*="css"] { font-family: 'Inter', system-ui, -apple-system, Segoe UI, Roboto, 'Helvetica Neue', Arial; }
section.main > div { padding-top: 1rem; }

/* gradient hero */
.hero {
  background: linear-gradient(135deg, #0ea5e9 0%, #8b5cf6 100%);
  color: white; border-radius: 20px; padding: 28px 28px;
  box-shadow: 0 8px 30px rgba(27,31,35,.15);
}

/* cards */
.card {
  background: white; border-radius: 16px; padding: 18px 18px;
  box-shadow: 0 10px 30px rgba(0,0,0,.06); border: 1px solid rgba(0,0,0,.04);
}

/* KPI cards */
.kpi { border-radius: 16px; padding: 14px 16px; background: #f8fafc; border: 1px solid #e5e7eb; }
.kpi .label { font-size: 0.92rem; color: #475569; }
.kpi .value { font-size: 1.35rem; font-weight: 700; color: #0f172a; }
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)

# -------------------- Helpers --------------------
@st.cache_data(show_spinner=False)
def download_to_tmp(url: str) -> str:
    import tempfile, os, requests
    ext = ".csv" if url.endswith(".csv") else ".parquet"
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

def safe_to_datetime(s):
    dt = pd.to_datetime(s, errors="coerce", format="%b-%Y")
    if dt.isna().all(): dt = pd.to_datetime(s, errors="coerce")
    return dt

# -------------------- Sidebar (data) --------------------
with st.sidebar:
    st.markdown("### Data Source")
    mode = st.radio("Load from", ["URL", "Upload"], horizontal=True)
    default_url = "https://github.com/altyn02/lending_club/releases/download/lending/accepted_merged.csv"
    url = st.text_input("Dataset URL", value=default_url) if mode == "URL" else None
    upload = st.file_uploader("Upload CSV / Parquet", type=["csv","parquet"]) if mode == "Upload" else None

    st.markdown("---")
    st.markdown("### Performance")
    sample_n = st.slider("Rows to analyze (sample)", 50_000, 500_000, 150_000, 50_000)
    st.caption("Sampling keeps the dashboard responsive.")

# -------------------- Load & light typing --------------------
if (mode == "URL" and not url) or (mode == "Upload" and upload is None):
    st.info("Add a URL or upload a file to start.")
    st.stop()

df_full = load_df(upload, url)

# Light type fixes
if "int_rate" in df_full and not pd.api.types.is_numeric_dtype(df_full["int_rate"]):
    df_full["int_rate"] = (
        df_full["int_rate"].astype(str).str.replace("%","", regex=False)
        .str.extract(r"([-+]?\d*\.?\d+)", expand=False).astype(float)
    )
if "revol_util" in df_full and not pd.api.types.is_numeric_dtype(df_full["revol_util"]):
    df_full["revol_util"] = (
        df_full["revol_util"].astype(str).str.replace("%","", regex=False)
        .str.extract(r"([-+]?\d*\.?\d+)", expand=False).astype(float)
    )
if "issue_d" in df_full:
    df_full["issue_d"] = safe_to_datetime(df_full["issue_d"])
    if df_full["issue_d"].notna().any():
        df_full["issue_year"] = df_full["issue_d"].dt.year

# Sample for speed
df = df_full.sample(min(len(df_full), sample_n), random_state=42) if len(df_full) > sample_n else df_full.copy()

# -------------------- HERO TITLE --------------------
TITLE = "Lending Club Credit Dashboard"
SUBTITLE = "Clean, minimal EDA for your merged dataset — fast overview & key signals."

st.markdown(
    f"""
    <div class="hero">
      <div style="display:flex; align-items:center; gap:16px;">
        <div style="font-size:2.0rem;line-height:1; font-weight:800;">💳 {TITLE}</div>
      </div>
      <div style="opacity:.95; margin-top:6px; font-size:0.98rem;">
        {SUBTITLE}
      </div>
    </div>
    """,
    unsafe_allow_html=True
)
st.write("")

# -------------------- KPI ROW --------------------
total_rows = len(df_full)
total_cols = df_full.shape[1]
date_min = df_full["issue_d"].min().date().isoformat() if "issue_d" in df_full and df_full["issue_d"].notna().any() else "—"
date_max = df_full["issue_d"].max().date().isoformat() if "issue_d" in df_full and df_full["issue_d"].notna().any() else "—"
target_col = "loan_status" if "loan_status" in df_full.columns else None
bad_ratio = "—"
if target_col:
    lc = df_full[target_col].astype(str).str.lower()
    bad = lc.isin(["charged off","default","late (31-120 days)"]).mean()
    bad_ratio = f"{bad*100:.1f}%"

k1, k2, k3, k4 = st.columns(4)
with k1:
    st.markdown(f'<div class="kpi"><div class="label">Total Rows</div><div class="value">{total_rows:,}</div></div>', unsafe_allow_html=True)
with k2:
    st.markdown(f'<div class="kpi"><div class="label">Columns</div><div class="value">{total_cols}</div></div>', unsafe_allow_html=True)
with k3:
    st.markdown(f'<div class="kpi"><div class="label">Issue Date Range</div><div class="value">{date_min} → {date_max}</div></div>', unsafe_allow_html=True)
with k4:
    st.markdown(f'<div class="kpi"><div class="label">Bad Rate (approx)</div><div class="value">{bad_ratio}</div></div>', unsafe_allow_html=True)

st.write("")

# -------------------- DASHBOARD GRID --------------------
# Row 1: Loans by Year (left) | Interest Rate (right)
r1c1, r1c2 = st.columns((1.25, 1))
with r1c1:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("#### Loans by Year")
    if "issue_year" in df.columns:
        year_counts = (
            df.dropna(subset=["issue_year"])
              .groupby("issue_year", as_index=False)
              .size().rename(columns={"size":"count"})
        )
        chart = alt.Chart(year_counts).mark_bar().encode(
            x=alt.X("issue_year:O", title="Year"),
            y=alt.Y("count:Q", title="Loans"),
            tooltip=["issue_year","count"]
        ).properties(height=320)
        st.altair_chart(chart, use_container_width=True)
    else:
        st.caption("No issue year found.")
    st.markdown('</div>', unsafe_allow_html=True)

with r1c2:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("#### Interest Rate Distribution")
    if "int_rate" in df.columns:
        hist = alt.Chart(df).mark_bar().encode(
            x=alt.X("int_rate:Q", bin=alt.Bin(maxbins=40), title="Interest Rate (%)"),
            y=alt.Y("count():Q", title="Count"),
            tooltip=["count()"]
        ).properties(height=320)
        st.altair_chart(hist, use_container_width=True)
    else:
        st.caption("Column 'int_rate' not found.")
    st.markdown('</div>', unsafe_allow_html=True)

# Row 2: Status by Grade (left) | Missingness (right)
r2c1, r2c2 = st.columns((1.25, 1))
with r2c1:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("#### Status by Grade")
    if {"loan_status","grade"}.issubset(df.columns):
        pv = pd.pivot_table(df, index="grade", columns="loan_status", aggfunc="size", fill_value=0)
        pv = pv.reset_index().melt(id_vars="grade", var_name="status", value_name="count")
        bar = alt.Chart(pv).mark_bar().encode(
            x=alt.X("grade:N", sort=alt.SortField("grade", order="ascending")),
            y="count:Q",
            color="status:N",
            tooltip=["grade","status","count"]
        ).properties(height=320)
        st.altair_chart(bar, use_container_width=True)
    else:
        st.caption("Need columns: loan_status & grade.")
    st.markdown('</div>', unsafe_allow_html=True)

with r2c2:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("#### Missingness (Top 12)")
    miss = df.isna().mean().sort_values(ascending=False).head(12).rename("missing_rate").reset_index()
    miss["missing_%"] = (miss["missing_rate"]*100).round(1)
    miss = miss.rename(columns={"index":"column"})
    miss_chart = alt.Chart(miss).mark_bar().encode(
        x=alt.X("missing_%:Q", title="Missing (%)"),
        y=alt.Y("column:N", sort="-x", title=""),
        tooltip=["column","missing_%"]
    ).properties(height=320)
    st.altair_chart(miss_chart, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

# -------------------- Footer --------------------
st.write("")
st.markdown(
    """
    <div style="text-align:center; color:#64748b; font-size:.9rem; padding:10px 0 0 0;">
      Design-first dashboard layout. Add modeling tabs later if needed.
    </div>
    """,
    unsafe_allow_html=True
)
