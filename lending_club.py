# app.py — Lending Club Dashboard (target-only, interactive filters)
# Dataset: early_pool_balanced_15k_each.csv

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
  margin-bottom: 12px;
}

.kpi { border-radius: 16px; padding: 14px 16px; background: #f8fafc; border: 1px solid #e5e7eb; }
.kpi .label { font-size: 0.92rem; color: #475569; }
.kpi .value { font-size: 1.35rem; font-weight: 700; color: #0f172a; }
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)

# -------------------- HERO --------------------
TITLE = "Lending Club Credit Dashboard"
SUBTITLE = "Explore distributions, relationships, and correlations in your balanced early-pool sample."
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

# -------------------- Light typing/cleanup --------------------
def to_float_pct(series: pd.Series) -> pd.Series:
    s = series.astype(str).str.replace("%","", regex=False).str.replace(",","", regex=False).str.strip()
    s = s.str.extract(r"([-+]?\d*\.?\d+)", expand=False)
    return pd.to_numeric(s, errors="coerce")

# Convert common percent-like columns if needed
for col in ["int_rate", "revol_util", "dti"]:
    if col in df_full and not pd.api.types.is_numeric_dtype(df_full[col]):
        df_full[col] = to_float_pct(df_full[col])

# Parse issue date if present (for filtering)
if "issue_d" in df_full.columns:
    issue_dt = pd.to_datetime(df_full["issue_d"], errors="coerce", format="%b-%Y")
    if issue_dt.isna().all():
        issue_dt = pd.to_datetime(df_full["issue_d"], errors="coerce")
    df_full["issue_d"] = issue_dt
    if df_full["issue_d"].notna().any():
        df_full["issue_year"] = df_full["issue_d"].dt.year

# -------------------- Sidebar Filters --------------------
with st.sidebar:
    st.subheader("Filters")

    # Date range filter (if issue_d exists and has values)
    if "issue_d" in df_full.columns and df_full["issue_d"].notna().any():
        dmin = pd.to_datetime(df_full["issue_d"].min()).date()
        dmax = pd.to_datetime(df_full["issue_d"].max()).date()
        date_range = st.date_input(
            "Issue date range",
            value=(dmin, dmax)
        )
    else:
        date_range = None
        st.caption("No issue date column found — charts won’t filter by date.")

    # Optional category filters if present
    grade_sel = None
    if "grade" in df_full.columns:
        opts = sorted(df_full["grade"].dropna().unique().tolist())
        grade_sel = st.multiselect("Grade", options=opts, default=[])

    term_sel = None
    if "term" in df_full.columns:
        term_opts = df_full["term"].astype(str).dropna().unique().tolist()
        term_sel = st.multiselect("Term", options=term_opts, default=[])

# -------------------- Apply Filters --------------------
df = df_full.copy()

if date_range and isinstance(date_range, (list, tuple)) and len(date_range) == 2:
    start_d, end_d = pd.to_datetime(date_range[0]), pd.to_datetime(date_range[1])
    if "issue_d" in df.columns:
        df = df[(df["issue_d"] >= start_d) & (df["issue_d"] <= end_d)]

if grade_sel is not None and len(grade_sel) > 0:
    df = df[df["grade"].isin(grade_sel)]

if term_sel is not None and len(term_sel) > 0:
    df = df[df["term"].astype(str).isin(term_sel)]

# -------------------- KPIs (no Issue Date Range KPI) --------------------
total_rows = len(df)
total_cols = df.shape[1]
bad_ratio = "—"
if "target" in df.columns:
    bad = (df["target"] == 1).mean()
    bad_ratio = f"{bad*100:.1f}%"

k1, k2, k3 = st.columns(3)
with k1:
    st.markdown(f'<div class="kpi"><div class="label">Filtered Rows</div><div class="value">{total_rows:,}</div></div>', unsafe_allow_html=True)
with k2:
    st.markdown(f'<div class="kpi"><div class="label">Columns</div><div class="value">{total_cols}</div></div>', unsafe_allow_html=True)
with k3:
    st.markdown(f'<div class="kpi"><div class="label">Bad Rate (target==1)</div><div class="value">{bad_ratio}</div></div>', unsafe_allow_html=True)

st.write("")

# -------------------- Tabs --------------------
tab_hist, tab_box, tab_density, tab_corr = st.tabs([
    "📊 Histograms", "📦 Boxplots", "🌫️ Density (KDE)", "🧮 Correlation Heatmap"
])

# ========== Histograms ==========
with tab_hist:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("Histogram")
    num_cols = [c for c in df.select_dtypes(include=[np.number]).columns if c != "target"]
    if len(num_cols) == 0:
        st.info("No numeric columns available.")
    else:
        col = st.selectbox("Numeric column", options=num_cols, index=min(0, len(num_cols)-1))
        bins = st.slider("Bins", 10, 80, 40, 5)
        # Overlay by target if present
        if "target" in df.columns:
            chart = alt.Chart(df).mark_bar(opacity=0.7).encode(
                x=alt.X(f"{col}:Q", bin=alt.Bin(maxbins=bins), title=col),
                y=alt.Y("count():Q", title="Count"),
                color=alt.Color("target:N", title="target"),
                tooltip=[col, "count()"]
            ).properties(height=340)
        else:
            chart = alt.Chart(df).mark_bar().encode(
                x=alt.X(f"{col}:Q", bin=alt.Bin(maxbins=bins), title=col),
                y=alt.Y("count():Q", title="Count"),
                tooltip=[col, "count()"]
            ).properties(height=340)
        st.altair_chart(chart, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

# ========== Boxplots ==========
with tab_box:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("Boxplot")
    if len(num_cols) == 0:
        st.info("No numeric columns available.")
    else:
        y_col = st.selectbox("Y (numeric)", options=num_cols, index=min(0, len(num_cols)-1), key="box_y")
        # X = target or a categorical if available
        cat_options = []
        if "target" in df.columns:
            cat_options.append("target")
        cat_options += [c for c in df.columns if df[c].dtype == "object" or df[c].dtype.name == "category"]
        cat_options = list(dict.fromkeys(cat_options))  # unique preserve order

        if len(cat_options) == 0:
            st.info("No categorical column (or target) to group by.")
        else:
            x_col = st.selectbox("X (category)", options=cat_options, index=0, key="box_x")
            box = alt.Chart(df).mark_boxplot().encode(
                x=alt.X(f"{x_col}:N", title=x_col),
                y=alt.Y(f"{y_col}:Q", title=y_col),
                color=alt.Color(f"{x_col}:N", legend=None)
            ).properties(height=340)
            st.altair_chart(box, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

# ========== Density (KDE via transform_density) ==========
with tab_density:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("Density (KDE)")
    if len(num_cols) == 0:
        st.info("No numeric columns available.")
    else:
        dens_col = st.selectbox("Numeric column", options=num_cols, index=min(0, len(num_cols)-1), key="dens")
        if "target" in df.columns:
            # Density by target
            dens = alt.Chart(df).transform_density(
                dens_col, groupby=["target"], as_=[dens_col, "density"]
            ).mark_area(opacity=0.5).encode(
                x=alt.X(f"{dens_col}:Q", title=dens_col),
                y=alt.Y("density:Q"),
                color="target:N"
            ).properties(height=340)
        else:
            dens = alt.Chart(df).transform_density(
                dens_col, as_=[dens_col, "density"]
            ).mark_area(opacity=0.6).encode(
                x=alt.X(f"{dens_col}:Q", title=dens_col),
                y=alt.Y("density:Q")
            ).properties(height=340)
        st.altair_chart(dens, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

# ========== Correlation Heatmap ==========
with tab_corr:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("Correlation Heatmap")
    # default features: top 10 by absolute correlation to target (if present)
    num_df = df.select_dtypes(include=[np.number]).copy()
    if num_df.empty:
        st.info("No numeric columns to correlate.")
    else:
        default_features = list(num_df.columns)
        if "target" in num_df.columns and len(num_df.columns) > 1:
            corr_to_target = num_df.corr(numeric_only=True)["target"].abs().sort_values(ascending=False)
            # drop target itself, take next 9 top features + target
            top_feats = [c for c in corr_to_target.index if c != "target"][:9]
            default_features = ["target"] + top_feats

        chosen = st.multiselect(
            "Select features for heatmap",
            options=list(num_df.columns),
            default=default_features
        )
        if len(chosen) < 2:
            st.info("Pick at least two features.")
        else:
            cmat = num_df[chosen].corr(numeric_only=True)
            corr_df = cmat.reset_index().melt("index")
            corr_df.columns = ["feature_x", "feature_y", "corr"]
            heat = alt.Chart(corr_df).mark_rect().encode(
                x=alt.X("feature_x:O", title="", sort=chosen),
                y=alt.Y("feature_y:O", title="", sort=chosen),
                color=alt.Color("corr:Q", scale=alt.Scale(scheme="blueorange", domain=[-1,1])),
                tooltip=["feature_x","feature_y", alt.Tooltip("corr:Q", format=".2f")]
            ).properties(height=420)
            st.altair_chart(heat, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

# -------------------- Footer --------------------
st.write("")
st.markdown(
    """
    <div style="text-align:center; color:#64748b; font-size:.9rem; padding:10px 0 0 0;">
      Interactive Streamlit dashboard • Filters drive all charts • Target-only dataset ✅
    </div>
    """,
    unsafe_allow_html=True
)
