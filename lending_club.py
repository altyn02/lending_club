# app.py — Lending Club Dashboard (auto-load from local or GitHub)
# Uses: early_pool_balanced_15k_each.csv
# Missingness section removed (dataset is clean)

import os
import tempfile
from pathlib import Path
import requests
import numpy as np
import pandas as pd
import streamlit as st
import altair as alt

# -------------------- Page & Theme --------------------
st.set_page_config(page_title="Lending Club Dashboard", page_icon="💳", layout="wide")
alt.data_transformers.disable_max_rows()
pd.set_option("display.max_columns", 200)

# -------------------- CSS (Design) --------------------
CSS = """
<style>
html, body, [class*="css"] { font-family: 'Inter', system-ui, -apple-system, Segoe UI, Roboto, 'Helvetica Neue', Arial; }
section.main > div { padding-top: 1rem; }

.hero {
  background: linear-gradient(135deg, #0ea5e9 0%, #8b5cf6 100%);
  color: white; border-radius: 20px; padding: 24px 24px;
  box-shadow: 0 8px 30px rgba(27,31,35,.15);
}

.card {
  background: white; border-radius: 16px; padding: 18px 18px;
  box-shadow: 0 10px 30px rgba(0,0,0,.06); border: 1px solid rgba(0,0,0,.04);
}

.kpi { border-radius: 16px; padding: 14px 16px; background: #f8fafc; border: 1px solid #e5e7eb; }
.kpi .label { font-size: 0.92rem; color: #475569; }
.kpi .value { font-size: 1.35rem; font-weight: 700; color: #0f172a; }
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)

# -------------------- HERO --------------------
TITLE = "Lending Club Credit Dashboard"
SUBTITLE = "Welcome 👋 — Explore, analyze, and visualize Lending Club data interactively."
LOGO_URL = "https://github.com/altyn02/lending_club/releases/download/lending_photo/lending.webp"

st.markdown(
    f"""
    <div class="hero">
      <div style="display:flex; align-items:center; gap:20px; flex-wrap:wrap;">
        <img src="{LOGO_URL}" alt="Logo" style="height:56px; border-radius:8px;">
        <div>
          <div style="font-size:2rem;font-weight:800;line-height:1.2;">{TITLE}</div>
          <div style="opacity:.95; margin-top:6px; font-size:1.05rem;">{SUBTITLE}</div>
        </div>
      </div>
    </div>
    """,
    unsafe_allow_html=True
)
st.write("")

# -------------------- Load dataset (local first, then GitHub fallback) --------------------
DATA_PATH = Path(__file__).with_name("early_pool_balanced_15k_each.csv")
DATA_URL = "https://github.com/altyn02/lending_club/releases/download/15k_lending/early_pool_balanced_15k_each.csv"

@st.cache_data(show_spinner=True)
def _download_csv_to_tmp(url: str) -> str:
    fd, tmp = tempfile.mkstemp(suffix=".csv"); os.close(fd)
    with requests.get(url, stream=True, timeout=300) as r:
        r.raise_for_status()
        with open(tmp, "wb") as f:
            for chunk in r.iter_content(chunk_size=1_048_576):
                if chunk:
                    f.write(chunk)
    return tmp

@st.cache_data(show_spinner=True)
def load_data(local_path: Path, url: str) -> pd.DataFrame:
    if local_path.exists():
        return pd.read_csv(local_path, low_memory=False)
    tmp = _download_csv_to_tmp(url)
    return pd.read_csv(tmp, low_memory=False)

df_full = load_data(DATA_PATH, DATA_URL)

# -------------------- Light Cleaning --------------------
# Convert percent-like columns if needed
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

# Parse issue date if present
if "issue_d" in df_full:
    issue_dt = pd.to_datetime(df_full["issue_d"], errors="coerce", format="%b-%Y")
    if issue_dt.isna().all():
        issue_dt = pd.to_datetime(df_full["issue_d"], errors="coerce")
    df_full["issue_d"] = issue_dt
    if df_full["issue_d"].notna().any():
        df_full["issue_year"] = df_full["issue_d"].dt.year

# Sampling (dataset is small/balanced, but keep pattern)
SAMPLE_N = 200_000
df = df_full.sample(min(len(df_full), SAMPLE_N), random_state=42)

# -------------------- KPIs --------------------
total_rows = len(df_full)
total_cols = df_full.shape[1]
date_min = df_full["issue_d"].min().date().isoformat() if "issue_d" in df_full and df_full["issue_d"].notna().any() else "—"
date_max = df_full["issue_d"].max().date().isoformat() if "issue_d" in df_full and df_full["issue_d"].notna().any() else "—"

# Target detection: prefer loan_status if present; else try 'target' 0/1
bad_ratio = "—"
if "loan_status" in df_full.columns:
    lc = df_full["loan_status"].astype(str).str.lower()
    bad = lc.isin(["charged off","default","late (31-120 days)"]).mean()
    bad_ratio = f"{bad*100:.1f}%"
elif "target" in df_full.columns:
    # Generic: treat 1 as "bad" if distribution looks balanced (your sample is balanced 15k each)
    try:
        bad = (df_full["target"] == 1).mean()
        bad_ratio = f"{bad*100:.1f}%"
    except Exception:
        pass

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

# -------------------- Dashboard Visuals --------------------
# Row 1: Loans by Year | Interest Rate
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

# Row 2: Status by Grade | Top Purposes (replaces Missingness)
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
    elif "grade" in df.columns and "target" in df.columns:
        # If only target exists
        pv = df.groupby(["grade","target"]).size().reset_index(name="count")
        pv["status"] = pv["target"].map({1:"Bad", 0:"Good"}).fillna(pv["target"].astype(str))
        bar = alt.Chart(pv).mark_bar().encode(
            x=alt.X("grade:N", sort=alt.SortField("grade", order="ascending")),
            y="count:Q",
            color="status:N",
            tooltip=["grade","status","count"]
        ).properties(height=320)
        st.altair_chart(bar, use_container_width=True)
    else:
        st.caption("Need columns: grade + (loan_status or target).")
    st.markdown('</div>', unsafe_allow_html=True)

with r2c2:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    if "purpose" in df.columns:
        st.markdown("#### Top 10 Loan Purposes")
        top_p = df["purpose"].fillna("Unknown").value_counts().head(10).reset_index()
        top_p.columns = ["purpose","count"]
        purpose_chart = alt.Chart(top_p).mark_bar().encode(
            x=alt.X("purpose:N", sort='-y', title="Purpose"),
            y=alt.Y("count:Q", title="Loans"),
            tooltip=["purpose","count"]
        ).properties(height=320)
        st.altair_chart(purpose_chart, use_container_width=True)
    else:
        st.markdown("#### Term Distribution")
        if "term" in df.columns:
            term_counts = df["term"].astype(str).value_counts().reset_index()
            term_counts.columns = ["term","count"]
            term_chart = alt.Chart(term_counts).mark_bar().encode(
                x=alt.X("term:N", sort='-y', title="Term"),
                y=alt.Y("count:Q", title="Count"),
                tooltip=["term","count"]
            ).properties(height=320)
            st.altair_chart(term_chart, use_container_width=True)
        else:
            st.caption("No 'purpose' or 'term' column available.")
    st.markdown('</div>', unsafe_allow_html=True)

# -------------------- Footer --------------------
st.write("")
st.markdown(
    """
    <div style="text-align:center; color:#64748b; font-size:.9rem; padding:10px 0 0 0;">
      Dashboard built with Streamlit — auto-loads balanced early-pool dataset 📊
    </div>
    """,
    unsafe_allow_html=True
)
