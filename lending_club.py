# app.py (Frontend-only / Visualization Mock)
# --------------------------------------------------------------
# This Streamlit app is a FRONTEND MOCK that works WITHOUT any dataset.
# It generates a realistic, LendingClub‑style demo dataframe in memory
# and renders the full UI/UX for:
#   • Data Exploration (EDA visuals)
#   • Preprocessing configuration (UI only)
#   • Data partitioning (UI only)
#   • Modeling (UI scaffold only — disabled without data)
#   • Batch scoring (UI scaffold only — disabled without data)
# Replace the demo generator with your real data later.
# --------------------------------------------------------------

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import streamlit as st
import altair as alt
import matplotlib.pyplot as plt

st.set_page_config(page_title="LendingClub Credit Modeling (Frontend Mock)", layout="wide", page_icon="💳")
alt.data_transformers.disable_max_rows()

# ------------------------------
# Demo data generator (no upload required)
# ------------------------------
@st.cache_data
def make_demo_df(n: int = 6000, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    start = datetime(2016, 1, 1)
    dates = [start + timedelta(days=int(x)) for x in rng.integers(0, 365*5, size=n)]

    loan_amnt = rng.normal(15000, 6000, size=n).clip(1000, 40000)
    int_rate = rng.normal(12.5, 4.0, size=n).clip(5.0, 36.0)
    dti = np.abs(rng.normal(16, 8, size=n)).clip(0, 45)
    fico = rng.normal(690, 60, size=n).clip(550, 850)
    revol_util = np.abs(rng.normal(45, 25, size=n)).clip(0, 100)

    term = rng.choice(["36 months", "60 months"], size=n, p=[0.65, 0.35])
    home_ownership = rng.choice(["RENT", "MORTGAGE", "OWN", "OTHER"], size=n, p=[0.45,0.4,0.13,0.02])
    purpose = rng.choice(["debt_consolidation","credit_card","home_improvement","small_business","medical","vacation"], size=n)

    # Synthetic probability of bad outcome (default/charged off)
    logit = (
        0.015*(int_rate-10) + 0.02*(dti-15) + 0.012*(revol_util-40) - 0.01*(fico-700)/10
        + (term == "60 months")*0.25 + (home_ownership == "RENT")*0.1
    )
    p_bad = 1/(1+np.exp(-logit))
    y = (rng.uniform(size=n) < p_bad).astype(int)

    status = np.where(y==1, "Charged Off", "Fully Paid")

    df = pd.DataFrame({
        "issue_d": pd.to_datetime(dates),
        "loan_amnt": loan_amnt.round(0),
        "int_rate": np.round(int_rate, 2),
        "dti": np.round(dti, 2),
        "fico_range_high": np.round(fico),
        "revol_util": np.round(revol_util, 1),
        "term": term,
        "home_ownership": home_ownership,
        "purpose": purpose,
        "loan_status": status
    })
    return df

# Single source of truth for this mock
DF = make_demo_df()
TARGET = "loan_status"  # (0=good, 1=bad) — here stored as text for visuals

# ------------------------------
# Sidebar (frontend controls only)
# ------------------------------
with st.sidebar:
    st.title("💳 LC Frontend Mock")
    st.caption("No dataset required — using synthetic demo data")

    st.subheader("Target & Labeling")
    st.text_input("Target column", value=TARGET, disabled=True)
    st.selectbox("Positive class (bad)", ["Charged Off"], index=0, disabled=True)

    st.markdown("---")
    st.subheader("Preprocessing")
    scale_numeric = st.checkbox("Standardize numeric features", value=True)
    drop_cols = st.multiselect(
        "Columns to drop (UI only)",
        options=list(DF.columns),
        default=["purpose"],
    )

    st.markdown("---")
    st.subheader("Partitioning")
    split_method = st.selectbox("Split method", ["Random (Stratified)", "Time-based (by issue_d)"])
    test_size = st.slider("Test size", 0.1, 0.5, 0.2, 0.05)
    cutoff_date = st.date_input("Cutoff date (train ≤ cutoff)", value=pd.to_datetime("2019-01-01"))

    st.markdown("---")
    st.subheader("Modeling (UI only)")
    model_name = st.selectbox("Model", ["Logistic Regression", "Random Forest", "Gradient Boosting"], index=1)
    st.slider("Decision threshold", 0.05, 0.95, 0.5, 0.01)

# ------------------------------
# Layout Tabs
# ------------------------------
eda_tab, prep_tab, split_tab, model_tab, predict_tab = st.tabs([
    "1) 🧭 Explore", "2) 🧹 Preprocess", "3) ✂️ Partition", "4) 🤖 Model", "5) 🔮 Predict"
])

# ------------------------------
# 1) Explore (charts)
# ------------------------------
with eda_tab:
    st.subheader("Dataset Overview")
    c1, c2, c3 = st.columns([2, 2, 1])
    with c1:
        st.dataframe(DF.head(50))
    with c2:
        st.write("**Summary (numeric)**")
        st.dataframe(DF.describe(include=[np.number]).T)
    with c3:
        st.write("**Missingness (demo)**")
        miss = DF.isna().mean().sort_values(ascending=False).to_frame("missing_rate")
        st.dataframe(miss)

    st.markdown("---")
    st.subheader("Target Distribution")
    target_counts = DF[TARGET].value_counts(dropna=False).rename_axis(TARGET).reset_index(name='count')
    chart = alt.Chart(target_counts).mark_bar().encode(
        x=alt.X(f"{TARGET}:N", sort='-y'), y='count:Q', tooltip=[TARGET, 'count']
    ).properties(height=300)
    st.altair_chart(chart, use_container_width=True)

    st.markdown("---")
    st.subheader("Feature Explorer")
    feature = st.selectbox("Choose a feature", [c for c in DF.columns if c != TARGET])
    if feature:
        if pd.api.types.is_numeric_dtype(DF[feature]):
            hist = alt.Chart(DF).mark_bar().encode(
                x=alt.X(f"{feature}:Q", bin=alt.Bin(maxbins=40)),
                y='count()',
                color=alt.Color(f"{TARGET}:N"),
                tooltip=[feature, TARGET]
            ).properties(height=300)
            st.altair_chart(hist, use_container_width=True)
        else:
            bar = alt.Chart(DF).mark_bar().encode(
                x=alt.X(f"{feature}:N", sort='-y'),
                y='count()',
                color=alt.Color(f"{TARGET}:N"),
                tooltip=[feature, TARGET]
            ).properties(height=300)
            st.altair_chart(bar, use_container_width=True)

    st.markdown("---")
    st.subheader("Bivariate: int_rate vs loan_amnt (colored by status)")
    scatter = alt.Chart(DF.sample(min(len(DF), 3000), random_state=1)).mark_circle(size=35, opacity=0.6).encode(
        x='int_rate:Q', y='loan_amnt:Q', color=f'{TARGET}:N', tooltip=['int_rate','loan_amnt',TARGET]
    ).properties(height=350)
    st.altair_chart(scatter, use_container_width=True)

    st.markdown("---")
    st.subheader("Correlations (numeric)")
    num_df = DF.select_dtypes(include=[np.number]).copy()
    # Create a numeric 0/1 for status just for corr viz
    num_df["bad"] = (DF[TARGET] == "Charged Off").astype(int)
    corr = num_df.corr(numeric_only=True)
    top = corr["bad"].abs().sort_values(ascending=False).head(10).index
    fig, ax = plt.subplots(figsize=(7,5))
    im = ax.imshow(corr.loc[top, top], aspect='auto')
    ax.set_xticks(range(len(top)))
    ax.set_yticks(range(len(top)))
    ax.set_xticklabels(top, rotation=45, ha='right')
    ax.set_yticklabels(top)
    ax.set_title("Top correlations (demo)")
    st.pyplot(fig)

# ------------------------------
# 2) Preprocess (UI only)
# ------------------------------
with prep_tab:
    st.subheader("Preprocessing Configuration (UI only)")
    st.write("Select options in the sidebar. In this mock, we only preview the effects — no transformations are applied.")

    numeric_cols = DF.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = [c for c in DF.columns if c not in numeric_cols]
    c1, c2, c3 = st.columns(3)
    with c1:
        st.write("**Numeric features**", len(numeric_cols))
        st.code(numeric_cols)
    with c2:
        st.write("**Categorical features**", len(categorical_cols))
        st.code(categorical_cols)
    with c3:
        st.write("**Dropped (UI)**", len(drop_cols))
        st.code(drop_cols)

    st.info("In the production app, numeric features will be imputed and optionally standardized; categoricals will be imputed and one‑hot encoded.")

# ------------------------------
# 3) Partition (UI only)
# ------------------------------
with split_tab:
    st.subheader("Partitioning (UI only)")
    min_d, max_d = DF["issue_d"].min(), DF["issue_d"].max()
    st.write(f"**issue_d range:** {min_d.date()} → {max_d.date()}")

    if split_method.startswith("Random"):
        st.success(f"Random stratified split preview — test_size={test_size:.2f}")
    else:
        st.success(f"Time‑based split preview — cutoff={pd.to_datetime(cutoff_date).date()} (train ≤ cutoff)")

    st.caption("This is a mock; no split is executed. In the real app we would persist X_train/X_test, y_train/y_test to session state.")

# ------------------------------
# 4) Model (disabled in mock)
# ------------------------------
with model_tab:
    st.subheader("Modeling (Frontend only)")
    st.warning("This is a visualization-only mock. Training is disabled because no real dataset is loaded.")

    kpi = pd.DataFrame({
        "metric": ["ROC-AUC","PR-AUC","F1","Precision","Recall"],
        "value": [0.82, 0.56, 0.61, 0.58, 0.65]
    })
    kpi_chart = alt.Chart(kpi).mark_bar().encode(x=alt.X("value:Q", scale=alt.Scale(domain=[0,1])), y=alt.Y("metric:N", sort='-x'), tooltip=["metric","value"])
    st.altair_chart(kpi_chart, use_container_width=True)

    st.caption("Dummy KPIs above are illustrative. Hook up your sklearn/xgboost pipeline later to compute real metrics and curves.")

# ------------------------------
# 5) Predict (disabled in mock)
# ------------------------------
with predict_tab:
    st.subheader("Batch Inference (Frontend only)")
    st.info("UI scaffold for uploading a CSV and downloading predictions. Disabled in this mock.")
    st.file_uploader("Upload CSV for prediction (disabled)", type=["csv"], disabled=True)
    st.text_input("Optional ID column", value="id", disabled=True)
    st.button("Score & Download (disabled)", disabled=True)

st.markdown("""
<hr/>
<p style='font-size: 0.9rem'>
This is a frontend‑only mock using synthetic LendingClub‑style data. When your real dataset is ready, replace the demo generator with a file uploader and wire tabs 2–5 to your preprocessing, splitting, training, and inference code.
</p>
""")
