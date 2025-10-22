# app.py — Lending Club Dashboard (uses existing merged dataset; design-first + your analysis)
# ---------------------------------------------------------------------

!pip install matplotlib

import numpy as np
import pandas as pd
import streamlit as st
import altair as alt
import matplotlib.pyplot as plt
import seaborn as sns

# -------------------- Page & Theme --------------------
st.set_page_config(page_title="Lending Club Dashboard", page_icon="💳", layout="wide")
alt.data_transformers.disable_max_rows()
pd.set_option("display.max_columns", 200)
pd.set_option("display.float_format", "{:.2f}".format)

# -------------------- CSS (Design) --------------------
CSS = """
<style>
html, body, [class*="css"] { font-family: 'Inter', system-ui, -apple-system, Segoe UI, Roboto, Arial; }
section.main > div { padding-top: 0.6rem; }

/* gradient hero */
.hero {
  background: linear-gradient(135deg, #0ea5e9 0%, #8b5cf6 100%);
  color: white; border-radius: 20px; padding: 22px 22px;
  box-shadow: 0 8px 30px rgba(27,31,35,.12);
}

/* cards */
.card {
  background: white; border-radius: 16px; padding: 16px 16px;
  box-shadow: 0 10px 30px rgba(0,0,0,.06); border: 1px solid rgba(0,0,0,.05);
  margin-bottom: 12px;
}

/* KPI cards */
.kpi { border-radius: 16px; padding: 12px 14px; background: #f8fafc; border: 1px solid #e5e7eb; }
.kpi .label { font-size: 0.9rem; color: #475569; }
.kpi .value { font-size: 1.25rem; font-weight: 800; color: #0f172a; }
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)

# -------------------- HERO --------------------
TITLE = "Lending Club Credit Dashboard"
SUBTITLE = "Welcome 👋 — Explore, analyze, and visualize your merged Lending Club dataset."
LOGO_URL = "https://raw.githubusercontent.com/altyn02/lending_club/main/assets/lendingclub_logo.png"

st.markdown(
    f"""
    <div class="hero">
      <div style="display:flex; align-items:center; gap:16px; flex-wrap:wrap;">
        <img src="{LOGO_URL}" alt="Logo" style="height:52px; border-radius:8px;">
        <div>
          <div style="font-size:1.9rem;font-weight:800;line-height:1.2;">{TITLE}</div>
          <div style="opacity:.98; margin-top:6px; font-size:1.02rem;">{SUBTITLE}</div>
        </div>
      </div>
    </div>
    """,
    unsafe_allow_html=True
)
st.write("")

# -------------------- Config --------------------
DATA_PATH = "accepted_merged.csv"        # <- your already-merged dataset
SAMPLE_N  = 200_000                      # sample for speed; tweak if you like
RANDOM_SEED = 42

# -------------------- Helpers --------------------
def to_float_pct(series: pd.Series) -> pd.Series:
    s = series.astype(str).str.replace("%","", regex=False).str.replace(",","", regex=False).str.strip()
    s = s.str.extract(r"([-+]?\d*\.?\d+)", expand=False)
    return pd.to_numeric(s, errors="coerce")

def to_float(series: pd.Series) -> pd.Series:
    s = series.astype(str).str.replace(",","", regex=False).str.strip()
    s = s.str.extract(r"([-+]?\d*\.?\d+)", expand=False)
    return pd.to_numeric(s, errors="coerce")

def safe_issue_date(s: pd.Series) -> pd.Series:
    dt = pd.to_datetime(s, errors="coerce", format="%b-%Y")
    if dt.isna().all():
        dt = pd.to_datetime(s, errors="coerce")
    return dt

@st.cache_data(show_spinner=True)
def load_data(path: str) -> pd.DataFrame:
    return pd.read_csv(path, low_memory=False)

# -------------------- Load Data --------------------
df = load_data(DATA_PATH)

# -------------------- 1) Quick Structural Overview (your Step 1.5–1.6) --------------------
with st.container():
    k1, k2, k3, k4 = st.columns(4)
    with k1:
        st.markdown(f'<div class="kpi"><div class="label">Rows</div><div class="value">{len(df):,}</div></div>', unsafe_allow_html=True)
    with k2:
        st.markdown(f'<div class="kpi"><div class="label">Columns</div><div class="value">{df.shape[1]}</div></div>', unsafe_allow_html=True)
    with k3:
        st.markdown(f'<div class="kpi"><div class="label">Numeric Columns</div><div class="value">{df.select_dtypes(include=[np.number]).shape[1]}</div></div>', unsafe_allow_html=True)
    with k4:
        st.markdown(f'<div class="kpi"><div class="label">Object Columns</div><div class="value">{df.select_dtypes(include=["object"]).shape[1]}</div></div>', unsafe_allow_html=True)

st.markdown('<div class="card">', unsafe_allow_html=True)
st.markdown("##### Column dtypes")
st.dataframe(df.dtypes.reset_index().rename(columns={"index":"column",0:"dtype"}), use_container_width=True)
st.markdown('</div>', unsafe_allow_html=True)

st.markdown('<div class="card">', unsafe_allow_html=True)
st.markdown("##### Missing values (top 15)")
missing_summary = df.isnull().sum().sort_values(ascending=False).head(15).rename("missing_count").reset_index()
missing_summary["missing_%"] = (missing_summary["missing_count"]/len(df)*100).round(2)
missing_summary = missing_summary.rename(columns={"index":"column"})
st.dataframe(missing_summary, use_container_width=True)
st.markdown('</div>', unsafe_allow_html=True)

# Drop extremely high-missing or irrelevant (your list; safe if absent)
drop_cols = [
    "url", "member_id", "id",
    "orig_projected_additional_accrued_interest",
    "payment_plan_start_date", "hardship_type", "hardship_reason",
    "hardship_status", "deferral_term", "hardship_amount",
    "hardship_start_date", "hardship_end_date", "hardship_dpd",
    "hardship_length", "hardship_loan_status"
]
existing_drop_cols = [c for c in drop_cols if c in df.columns]
if existing_drop_cols:
    df.drop(columns=existing_drop_cols, inplace=True, errors="ignore")

# Show a small sample and numeric stats
st.markdown('<div class="card">', unsafe_allow_html=True)
c1, c2 = st.columns((1.4, 1))
with c1:
    st.markdown("##### Head (5)")
    st.dataframe(df.head(5), use_container_width=True)
with c2:
    st.markdown("##### Basic numeric statistics")
    st.dataframe(df.select_dtypes(include=[np.number]).describe().T.head(15), use_container_width=True)
st.markdown('</div>', unsafe_allow_html=True)

# -------------------- DTI cleaning (one-time) --------------------
if "dti" in df.columns:
    df["dti"] = to_float_pct(df["dti"]) if not pd.api.types.is_numeric_dtype(df["dti"]) else df["dti"]
    valid_min, valid_max = 0, 99
    flag_threshold = 70
    df.loc[(df["dti"] < valid_min) | (df["dti"] > valid_max), "dti"] = np.nan
    df["high_dti_flag"] = (df["dti"] > flag_threshold).astype("Int64")

# -------------------- Tabs: Explore | Status×Grade | Stats | Preprocess --------------------
tab1, tab2, tab3, tab4 = st.tabs(["🧭 Explore", "🏷️ Status × Grade", "📊 Descriptive Stats", "🧹 Preprocess"])

# ========== Tab 1: Explore (your Step 2.1 plus extras) ==========
with tab1:
    # Light parsing for dates & percents used in visuals
    if "int_rate" in df.columns and not pd.api.types.is_numeric_dtype(df["int_rate"]):
        df["int_rate"] = to_float_pct(df["int_rate"])
    if "revol_util" in df.columns and not pd.api.types.is_numeric_dtype(df["revol_util"]):
        df["revol_util"] = to_float_pct(df["revol_util"])
    if "issue_d" in df.columns:
        df["_issue_dt"] = safe_issue_date(df["issue_d"])
        df["issue_year"] = df["_issue_dt"].dt.year

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("Loans by Issue Year")
    if "issue_year" in df.columns:
        year_counts = (df.dropna(subset=["issue_year"])
                         .groupby("issue_year", as_index=False)
                         .size()
                         .rename(columns={"size":"count"}))
        if not year_counts.empty:
            chart = alt.Chart(year_counts).mark_bar().encode(
                x=alt.X("issue_year:O", title="Year"),
                y=alt.Y("count:Q", title="Loans"),
                tooltip=["issue_year","count"]
            ).properties(height=330)
            st.altair_chart(chart, use_container_width=True)

            table_year = year_counts.copy()
            table_year["share(%)"] = (table_year["count"]/table_year["count"].sum()*100).round(2)
            st.dataframe(table_year, use_container_width=True)
        else:
            st.info("No valid 'issue_d' values to compute yearly counts.")
    else:
        st.info("Column 'issue_d' not found.")
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="card">', unsafe_allow_html=True)
    c1, c2 = st.columns((1.1, 1))
    with c1:
        st.markdown("#### Interest Rate Distribution")
        if "int_rate" in df.columns:
            hist = alt.Chart(df).mark_bar().encode(
                x=alt.X("int_rate:Q", bin=alt.Bin(maxbins=40), title="Interest Rate (%)"),
                y=alt.Y("count():Q", title="Count"),
                tooltip=["count()"]
            ).properties(height=300)
            st.altair_chart(hist, use_container_width=True)
        else:
            st.caption("Missing column: int_rate")
    with c2:
        st.markdown("#### Top 10 Loan Purposes")
        if "purpose" in df.columns:
            top_p = df["purpose"].fillna("Unknown").value_counts().head(10).reset_index()
            top_p.columns = ["purpose","count"]
            bar = alt.Chart(top_p).mark_bar().encode(
                x=alt.X("purpose:N", sort='-y', title="Purpose"),
                y=alt.Y("count:Q", title="Loans"),
                tooltip=["purpose","count"]
            ).properties(height=300)
            st.altair_chart(bar, use_container_width=True)
        else:
            st.caption("Missing column: purpose")
    st.markdown('</div>', unsafe_allow_html=True)

# ========== Tab 2: Status × Grade (your Step 2.2) ==========
with tab2:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("Outcome distribution by credit grade")
    required_cols = {"loan_status","grade"}
    missing_cols = [c for c in required_cols if c not in df.columns]
    if missing_cols:
        st.warning(f"Missing columns: {missing_cols}")
    else:
        pivot_counts = (
            pd.pivot_table(df, index="loan_status", columns="grade", aggfunc="size", fill_value=0)
            .astype(int)
        )
        pivot_counts["Total"] = pivot_counts.sum(axis=1)
        pivot_perc = (pivot_counts.div(pivot_counts["Total"], axis=0) * 100).round(2)

        st.markdown("##### Counts")
        st.dataframe(pivot_counts, use_container_width=True)

        st.markdown("##### Percentages (%)")
        st.dataframe(pivot_perc, use_container_width=True)

        # Stacked bar of proportions by grade
        pv_long = pivot_perc.drop(columns=["Total"], errors="ignore").reset_index().melt(
            id_vars="loan_status", var_name="grade", value_name="pct"
        )
        chart = alt.Chart(pv_long).mark_bar().encode(
            x=alt.X("grade:N", title="Grade"),
            y=alt.Y("pct:Q", title="Percentage"),
            color=alt.Color("loan_status:N", title="Status"),
            tooltip=["grade","loan_status","pct"]
        ).properties(height=350)
        st.altair_chart(chart, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

# ========== Tab 3: Descriptive Statistics (your Step 2.3) ==========
with tab3:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("Descriptive statistics (numeric)")
    numeric_df = df.select_dtypes(include=[np.number])
    if not numeric_df.empty:
        desc_stats = numeric_df.describe().T
        desc_stats["missing_count"] = df[numeric_df.columns].isnull().sum()
        desc_stats["missing_%"] = (desc_stats["missing_count"] / len(df) * 100).round(2)
        desc_stats["coeff_var"] = (desc_stats["std"] / desc_stats["mean"]).replace([np.inf, -np.inf], np.nan).round(2)
        desc_stats = desc_stats[["count","mean","std","min","25%","50%","75%","max","missing_%","coeff_var"]].round(2)
        st.dataframe(desc_stats, use_container_width=True)
        st.caption(f"Displayed {desc_stats.shape[0]} numeric variables.")
    else:
        st.info("No numeric columns detected.")
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("Pairwise relationships (sample)")
    # Optional: a small seaborn grid (kept small for speed)
    cols_for_pairs = [c for c in ["loan_amnt","annual_inc","int_rate","dti"] if c in df.columns]
    if len(cols_for_pairs) >= 2:
        sample = df[cols_for_pairs].dropna().sample(min(2000, len(df)), random_state=RANDOM_SEED)
        fig = sns.pairplot(sample, diag_kind="kde")
        st.pyplot(fig)
        plt.close('all')
    else:
        st.caption("Need at least two of: loan_amnt, annual_inc, int_rate, dti")
    st.markdown('</div>', unsafe_allow_html=True)

# ========== Tab 4: Preprocess (your Step 3.* summarized as UI run) ==========
with tab4:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("Initial cleaning & target construction")

    # 3.1 Remove duplicates
    before = len(df)
    df = df.drop_duplicates()
    after = len(df)
    st.write(f"**Duplicates removed:** {before - after:,} (Remaining: {after:,})")

    # Normalize loan_status labels (legacy forms)
    if "loan_status" in df.columns:
        ls = df["loan_status"].astype(str).str.strip().str.lower()
        ls = ls.str.replace(r"does not meet the credit policy\. status:\s*charged off", "charged off", regex=True)
        ls = ls.str.replace(r"does not meet the credit policy\. status:\s*fully paid", "fully paid", regex=True)
        ls = ls.str.replace(r"\s+", " ", regex=True)
        df["loan_status"] = ls

        INCLUDE_LATE_31_120 = True
        mapping = {"fully paid": 1, "charged off": 0, "default": 0}
        if INCLUDE_LATE_31_120:
            mapping["late (31-120 days)"] = 0

        status_clean = df["loan_status"].astype(str).str.strip().str.lower()
        status_counts_before = status_clean.value_counts(dropna=False).sort_index()
        st.markdown("##### Loan status distribution (raw)")
        st.dataframe(status_counts_before.to_frame("count"))

        df = df.assign(target = status_clean.map(mapping))
        dropped_mask = df["target"].isna()
        dropped_status_counts = status_clean[dropped_mask].value_counts().sort_index()
        df = df.loc[~dropped_mask].copy()

        st.markdown("##### Dropped non-final/ambiguous statuses")
        if not dropped_status_counts.empty:
            st.dataframe(dropped_status_counts.to_frame("dropped_count"))
        st.write(f"**Rows kept for supervised learning:** {len(df):,}")

        st.markdown("##### Target distribution (share)")
        st.dataframe(df["target"].value_counts(normalize=True).round(3).to_frame("share"))
    else:
        st.warning("Column 'loan_status' not found — skipping target creation.")

    # 3.3 Drop columns with >40% missing (except essentials)
    THRESHOLD = 0.40
    essential_cols = {
        "loan_amnt","term","int_rate","installment","grade","sub_grade","emp_length",
        "home_ownership","annual_inc","verification_status","purpose","addr_state",
        "dti","revol_util","revol_bal","open_acc","total_acc",
        "fico_range_low","fico_range_high","issue_d","application_type","target"
    }
    miss_ratio = df.isnull().mean().sort_values(ascending=False).to_frame("missing_ratio")
    miss_ratio["missing_%"] = (miss_ratio["missing_ratio"]*100).round(2)
    st.markdown("##### Top 20 missing % BEFORE dropping")
    st.dataframe(miss_ratio.head(20))

    candidates = miss_ratio.index[miss_ratio["missing_ratio"] > THRESHOLD].tolist()
    to_drop = [c for c in candidates if c not in essential_cols]
    before_cols = df.shape[1]
    df = df.drop(columns=to_drop, errors="ignore").copy()
    after_cols = df.shape[1]

    st.write(f"**Columns dropped (>40% missing):** {before_cols - after_cols} — Remaining: {after_cols}")
    st.caption("Essentials are kept even if sparse (for modeling later).")

    # 3.4 Type & format fixes
    percent_cols = [c for c in ["int_rate","revol_util","dti"] if c in df.columns]
    for c in percent_cols:
        if not pd.api.types.is_numeric_dtype(df[c]):
            df[c] = to_float_pct(df[c])

    num_like_cols = [c for c in [
        "loan_amnt","funded_amnt","funded_amnt_inv","annual_inc","revol_bal",
        "installment","open_acc","total_acc","pub_rec","delinq_2yrs",
        "inq_last_6mths","mort_acc","pub_rec_bankruptcies","collections_12_mths_ex_med"
    ] if c in df.columns]
    for c in num_like_cols:
        if not pd.api.types.is_numeric_dtype(df[c]):
            df[c] = to_float(df[c])

    if {"fico_range_low","fico_range_high"}.issubset(df.columns):
        df["fico_mean"] = (
            pd.to_numeric(df["fico_range_low"], errors="coerce")
            + pd.to_numeric(df["fico_range_high"], errors="coerce")
        ) / 2.0

    if "term" in df.columns:
        df["term_months"] = df["term"].astype(str).str.extract(r"(\d+)", expand=False).astype(float)

    if "emp_length" in df.columns:
        s = df["emp_length"].astype(str).str.lower().str.strip()
        s = s.replace({"10+ years":"10","< 1 year":"0","n/a":np.nan}).str.extract(r"(\d+)", expand=False)
        df["emp_length_years"] = pd.to_numeric(s, errors="coerce")

    for c in ["issue_d","earliest_cr_line","last_credit_pull_d"]:
        if c in df.columns:
            df[c] = safe_issue_date(df[c])

    if "issue_d" in df.columns:
        df["issue_year"]  = df["issue_d"].dt.year
        df["issue_month"] = df["issue_d"].dt.month

    st.markdown("##### Dtypes (peek)")
    st.dataframe(df.dtypes.head(20).reset_index().rename(columns={"index":"column",0:"dtype"}), use_container_width=True)

# -------------------- Footer --------------------
st.write("")
st.markdown(
    """
    <div style="text-align:center; color:#64748b; font-size:.9rem; padding:12px 0 0 0;">
      Design-first Streamlit dashboard • Feel free to add modeling & prediction tabs next.
    </div>
    """,
    unsafe_allow_html=True
)
