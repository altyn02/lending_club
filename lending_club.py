# app.py — Lending Club Dashboard (fixed EDA vars, no density, Logit in its own tab)

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

CSS = """
<style>
html, body, [class*="css"] { font-family: 'Inter', system-ui, -apple-system, Segoe UI, Roboto, 'Helvetica Neue', Arial; }
section.main > div { padding-top: 1rem; }
.hero{background:linear-gradient(135deg,#0ea5e9 0%,#8b5cf6 100%);color:white;border-radius:20px;padding:24px;box-shadow:0 8px 30px rgba(27,31,35,.15)}
.card{background:white;border-radius:16px;padding:18px;box-shadow:0 10px 30px rgba(0,0,0,.06);border:1px solid rgba(0,0,0,.04);margin-bottom:12px}
.kpi{border-radius:16px;padding:14px 16px;background:#f8fafc;border:1px solid #e5e7eb}
.kpi .label{font-size:.92rem;color:#475569}.kpi .value{font-size:1.35rem;font-weight:700;color:#0f172a}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)

# -------------------- HERO --------------------
TITLE = "Lending Club Credit Dashboard"
SUBTITLE = "Explore distributions, correlations, and a simple logit model (fast & clean)"
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

for col in ["int_rate", "revol_util", "dti"]:
    if col in df_full and not pd.api.types.is_numeric_dtype(df_full[col]):
        df_full[col] = to_float_pct(df_full[col])

if "issue_year" not in df_full.columns:
    if "issue_d" in df_full.columns:
        issue_dt = pd.to_datetime(df_full["issue_d"], errors="coerce", format="%b-%Y")
        if issue_dt.isna().all():
            issue_dt = pd.to_datetime(df_full["issue_d"], errors="coerce")
        df_full["issue_year"] = issue_dt.dt.year

# -------------------- Sidebar Filters --------------------
with st.sidebar:
    st.subheader("Filters")

    year_range = None
    if "issue_year" in df_full.columns and df_full["issue_year"].notna().any():
        years = pd.to_numeric(df_full["issue_year"], errors="coerce").dropna().astype(int)
        min_year, max_year = int(years.min()), int(years.max())
        year_range = st.slider("Filter by Issue Year", min_value=min_year, max_value=max_year, value=(min_year, max_year))
    else:
        st.caption("No issue_year column found — charts won’t filter by year.")

    grade_sel = None
    if "grade" in df_full.columns:
        opts = sorted(pd.Series(df_full["grade"]).dropna().astype(str).unique().tolist())
        grade_sel = st.multiselect("Grade", options=opts, default=[])

    term_sel = None
    if "term" in df_full.columns:
        term_opts = pd.Series(df_full["term"]).dropna().astype(str).unique().tolist()
        term_sel = st.multiselect("Term", options=term_opts, default=[])

# -------------------- Apply Filters --------------------
df = df_full.copy()

if year_range and "issue_year" in df.columns:
    iy = pd.to_numeric(df["issue_year"], errors="coerce")
    df = df[(iy >= year_range[0]) & (iy <= year_range[1])]

if grade_sel:
    df = df[df["grade"].astype(str).isin(grade_sel)]
if term_sel:
    df = df[df["term"].astype(str).isin(term_sel)]

# Consistent target type for coloring
if "target" in df.columns:
    df["target"] = df["target"].astype("category")

# -------------------- KPIs --------------------
total_rows = len(df)
total_cols = df.shape[1]
bad_ratio = "—"
if "target" in df.columns:
    try:
        bad = (pd.to_numeric(df["target"], errors="coerce").fillna(0).astype(int) == 1).mean()
        bad_ratio = f"{bad*100:.1f}%"
    except Exception:
        pass

k1, k2, k3 = st.columns(3)
with k1:
    st.markdown(f'<div class="kpi"><div class="label">Filtered Rows</div><div class="value">{total_rows:,}</div></div>', unsafe_allow_html=True)
with k2:
    st.markdown(f'<div class="kpi"><div class="label">Columns</div><div class="value">{total_cols}</div></div>', unsafe_allow_html=True)
with k3:
    st.markdown(f'<div class="kpi"><div class="label">Bad Rate </div><div class="value">{bad_ratio}</div></div>', unsafe_allow_html=True)

st.write("")
# Global target legend (always visible)
st.markdown(
    """
    <div style="background:#f1f5f9;padding:10px 14px;border-radius:10px;font-size:0.95rem;">
      🎯 <b>Target legend:</b> <b>0</b> = Charged Off &nbsp;&nbsp;|&nbsp;&nbsp; <b>1</b> = Fully Paid
    </div>
    """,
    unsafe_allow_html=True
)
st.write("")

# -------------------- EDA variables (fixed) --------------------
from pandas.api.types import is_numeric_dtype

REQUIRED = ["loan_amnt", "int_rate", "delinq_2yrs", "annual_inc", "dti"]
# Use full filtered data for EDA (five vars only; safe and fast)
df_eda = df.copy()

EDA_VARS = [c for c in REQUIRED if c in df_eda.columns and is_numeric_dtype(df_eda[c])]
missing = [c for c in REQUIRED if c not in EDA_VARS]
if missing:
    st.warning("These requested variables are absent or non-numeric in the filtered data: " + ", ".join(missing))

# Helper for ranking (used by heatmap/pairwise defaults)
def get_featured_vars(df, k=6):
    numeric_pool = [c for c in df.select_dtypes(include=[np.number]).columns if c != "target"]
    target_num = None
    if "target" in df.columns:
        t = pd.to_numeric(df["target"], errors="coerce")
        if t.notna().sum() >= 2 and t.nunique(dropna=True) >= 2:
            target_num = t
    if target_num is not None and numeric_pool:
        tmp = pd.concat([target_num.rename("target_num"), df[numeric_pool]], axis=1).dropna()
        if not tmp.empty:
            cabs = tmp.corr(numeric_only=True)["target_num"].drop("target_num", errors="ignore").abs()
            top_num = cabs.sort_values(ascending=False).index.tolist()[:k]
        else:
            top_num = numeric_pool[:k]
    else:
        top_num = numeric_pool[:k]
    return top_num

def categorical_cols(df: pd.DataFrame, max_card: int = 30, include_target_if_cat: bool = True) -> list:
    """Return small-cardinality categorical-like columns (including low-cardinality numerics)."""
    cats = []
    for c in df.columns:
        if df[c].dtype.name in ("object", "category"):
            if df[c].dropna().nunique() <= max_card:
                cats.append(c)
        elif is_numeric_dtype(df[c]):
            u = df[c].dropna().nunique()
            if 2 <= u <= max_card:
                cats.append(c)
    if include_target_if_cat and "target" in df.columns:
        t = df["target"]
        if (t.dtype.name in ("object", "category")) or (is_numeric_dtype(t) and t.dropna().nunique() <= max_card):
            if "target" not in cats:
                cats = ["target"] + cats
    return list(dict.fromkeys(cats))

# -------------------- Tabs (Density removed; Logit in its own tab) --------------------
tab_data, tab_hist, tab_box, tab_corr, tab_ttest, tab_pair, tab_logit = st.tabs([
    "🧭 Data Exploration","📊 Histograms", "📦 Boxplots", "🧮 Correlation Heatmap", "⚖️ T-tests / ANOVA", "🔗 Pairwise (Sample)", "🧠 Logit"
])

# ========== Data Exploration ==========

with tab_data:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("Data Exploration — quick view")
    st.write("Sample of the dataframe used for visualizations (filters applied).")

    # sample for display (cap for the UI)
    SAMPLE_N = EDA_SAMPLE_N if 'EDA_SAMPLE_N' in globals() else 10000
    sample = df if len(df) <= SAMPLE_N else df.sample(SAMPLE_N, random_state=42)

    # quick metrics
    rows, cols = sample.shape
    missing_pct = sample.isna().mean().mean() * 100
    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("Rows (shown)", f"{rows:,}")
    with c2:
        st.metric("Columns", f"{cols}")
    with c3:
        st.metric("Avg missing", f"{missing_pct:.2f}%")

    st.markdown("#### Head (first rows)")
    st.dataframe(sample.head(10), use_container_width=True)

    st.markdown("#### Statistical summary (describe)")
    desc = sample.describe(include="all").T
    # format numeric-like columns to 3 decimals for readability
    for col in desc.columns:
        try:
            desc[col] = pd.to_numeric(desc[col], errors="coerce").round(3).combine_first(desc[col])
        except Exception:
            pass
    st.dataframe(desc, use_container_width=True)

    st.markdown('</div>', unsafe_allow_html=True)
    
# ========== Histograms (fixed variables, no chunking) ==========
with tab_hist:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("Histogram — Selected Variables")
    st.caption("🎯 Target legend: 0 = Charged Off, 1 = Fully Paid")

    if not EDA_VARS:
        st.info("No suitable numeric columns from the requested list.")
    else:
        bins = st.slider("Bins", 10, 80, 40, 5, key="hist_bins_selected")
        src = df_eda[EDA_VARS + (["target"] if "target" in df_eda.columns else [])].dropna()

        chart = (
            alt.Chart(src)
            .transform_fold(EDA_VARS, as_=["variable", "value"])
            .mark_bar(opacity=0.7)
            .encode(
                x=alt.X("value:Q", bin=alt.Bin(maxbins=bins), title=None),
                y=alt.Y("count():Q", title="Count"),
                color=alt.Color("target:N", title="target") if "target" in src.columns else alt.value(None),
                facet=alt.Facet("variable:N", columns=3, title="")
            )
            .properties(height=180)
        )
        st.altair_chart(chart, use_container_width=True)

    st.markdown('</div>', unsafe_allow_html=True)

# ========== Boxplots (fixed variables, grouped by target if available) ==========
with tab_box:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("Boxplot — Selected Variables")
    st.caption("🎯 Target legend: 0 = Charged Off, 1 = Fully Paid")

    if not EDA_VARS:
        st.info("No suitable numeric columns from the requested list.")
    else:
        # group by 'target' if present; otherwise first small-cardinality categorical
        if "target" in df_eda.columns:
            x_col = "target"
        else:
            cat_cols = [c for c in df_eda.columns
                        if df_eda[c].dtype.name in ("object", "category")
                        and df_eda[c].dropna().nunique() <= 30]
            x_col = cat_cols[0] if cat_cols else None

        if not x_col:
            st.info("No categorical column to group by.")
        else:
            src = df_eda[[x_col] + EDA_VARS].dropna()
            melt = src.melt(id_vars=[x_col], value_vars=EDA_VARS,
                            var_name="variable", value_name="value")
            chart = (
                alt.Chart(melt)
                .mark_boxplot()
                .encode(
                    x=alt.X(f"{x_col}:N", title=x_col),
                    y=alt.Y("value:Q", title=None),
                    color=alt.Color(f"{x_col}:N", legend=None),
                    facet=alt.Facet("variable:N", columns=3, title="")
                )
                .properties(height=180)
            )
            st.altair_chart(chart, use_container_width=True)

    st.markdown('</div>', unsafe_allow_html=True)

# ========== Correlation Heatmap (kept; lightweight) ==========
with tab_corr:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("Correlation Heatmap (numeric only)")

    num_df = df.select_dtypes(include=[np.number]).copy()
    if num_df.empty or num_df.shape[1] < 2:
        st.info("Not enough numeric columns for a correlation heatmap.")
    else:
        # Prefer your selected variables + target if numeric
        defaults = [c for c in EDA_VARS if c in num_df.columns]
        ordered = defaults[:]
        if "target" in num_df.columns and "target" not in ordered:
            ordered = ["target"] + ordered

        # if we still have <2, fall back to auto
        if len(ordered) < 2:
            ordered = get_featured_vars(num_df, k=min(8, num_df.shape[1]))

        cmat = num_df[ordered].corr(numeric_only=True)
        corr_df = cmat.reset_index().melt("index")
        corr_df.columns = ["feature_x", "feature_y", "corr"]

        heat = alt.Chart(corr_df).mark_rect().encode(
            x=alt.X("feature_x:O", title="", sort=ordered),
            y=alt.Y("feature_y:O", title="", sort=ordered),
            color=alt.Color("corr:Q", scale=alt.Scale(scheme="blueorange", domain=[-1, 1])),
            tooltip=["feature_x", "feature_y", alt.Tooltip("corr:Q", format=".2f")]
        ).properties(height=420)
        st.altair_chart(heat, use_container_width=True)

    st.markdown('</div>', unsafe_allow_html=True)

# ========== T- TEST ==========

with tab_ttest:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("T-tests & ANOVA — quick comparisons")
    st.write("Compare numeric features across groups (default: target). Uses Welch t-test for 2 groups and one-way ANOVA for >2 groups. Results can be downloaded as CSV.")

    try:
        from scipy import stats
    except Exception as e:
        st.info("scipy is required for this tab. Add `scipy` to requirements.")
        st.exception(e)
        st.markdown('</div>', unsafe_allow_html=True)
        st.stop()

    # get candidate grouping columns; ensure 'target' is available if it's the only categorical-like column
    group_opts = categorical_cols(df_eda, max_card=50, include_target_if_cat=True) or []
    if not group_opts and "target" in df_eda.columns:
        group_opts = ["target"]

    if not group_opts:
        st.info("No categorical-like column available for grouping.")
    else:
        group_col = st.selectbox("Group by (categorical)", options=group_opts, index=0 if "target" in group_opts else 0)
        # numeric candidates (exclude grouping col)
        num_candidates = [c for c in df_eda.select_dtypes(include=[np.number]).columns if c != group_col]
        if not num_candidates:
            st.info("No numeric columns to test.")
        else:
            cap = st.slider("Max features to test (cap)", 3, min(60, max(3, len(num_candidates))), min(12, len(num_candidates)))
            preset = st.radio("Select features", ["Top by |corr with grouping (if binary)|", "Manual"], horizontal=True)

            if preset.startswith("Top"):
                unique_vals = df_eda[group_col].dropna().unique()
                if len(unique_vals) == 2:
                    try:
                        grpnum = pd.to_numeric(df_eda[group_col].astype('category').cat.codes, errors="coerce")
                        corr_abs = df_eda[num_candidates].corrwith(grpnum, numeric_only=True).abs().sort_values(ascending=False)
                        chosen = corr_abs.index.tolist()[:cap]
                    except Exception:
                        chosen = num_candidates[:cap]
                else:
                    chosen = num_candidates[:cap]
            else:
                chosen = st.multiselect("Pick numeric features", options=num_candidates, default=num_candidates[:min(cap, len(num_candidates))])

            chosen = [c for c in chosen if c in df_eda.columns][:cap]
            if not chosen:
                st.info("No features selected.")
            else:
                # robust sample slider (handles missing EDA_SAMPLE_N)
                eda_sample = globals().get("EDA_SAMPLE_N", 10000)
                try:
                    eda_sample = int(eda_sample)
                except Exception:
                    eda_sample = 10000

                max_sample = max(1, min(len(df), eda_sample))
                min_value = 1
                default = min(max_sample, 5000)
                step = 500 if max_sample >= 500 else 1
                sample_n = st.slider("Sample rows for speed", min_value=min_value, max_value=max_sample, value=default, step=step)

                src = df.sample(sample_n, random_state=42) if len(df) > sample_n else df.copy()
                src = src[[group_col] + chosen].dropna()

                @st.cache_data
                def compute_tests(src_df, group_col, features):
                    rows = []
                    groups = src_df[group_col].astype("category")
                    levels = list(groups.cat.categories)
                    for f in features:
                        row = {"feature": f}
                        vals = []
                        for lv in levels:
                            arr = pd.to_numeric(src_df.loc[groups == lv, f], errors="coerce").dropna().values
                            vals.append(arr)
                        if len(vals) >= 2 and all(len(v) >= 2 for v in vals):
                            if len(vals) == 2:
                                t_eq, p_eq = stats.ttest_ind(vals[0], vals[1], equal_var=True)
                                t_w, p_w = stats.ttest_ind(vals[0], vals[1], equal_var=False)
                                m1, m2 = vals[0].mean(), vals[1].mean()
                                s1, s2 = vals[0].std(ddof=1), vals[1].std(ddof=1)
                                denom = np.sqrt((s1**2 + s2**2) / 2) if (s1 > 0 or s2 > 0) else np.nan
                                cohens_d = (m1 - m2) / denom if denom and not np.isnan(denom) else np.nan
                                row.update({
                                    "F_statistic": np.nan, "F_pvalue": np.nan,
                                    "t_equal": float(t_eq), "p_equal": float(p_eq),
                                    "t_welch": float(t_w), "p_welch": float(p_w),
                                    "t_final": float(t_w), "p_final": float(p_w),
                                    "cohens_d": float(cohens_d)
                                })
                            else:
                                try:
                                    F, pF = stats.f_oneway(*vals)
                                except Exception:
                                    F, pF = np.nan, np.nan
                                row.update({
                                    "F_statistic": float(F), "F_pvalue": float(pF),
                                    "t_equal": np.nan, "p_equal": np.nan,
                                    "t_welch": np.nan, "p_welch": np.nan,
                                    "t_final": np.nan, "p_final": np.nan,
                                    "cohens_d": np.nan
                                })
                        else:
                            row.update({
                                "F_statistic": np.nan, "F_pvalue": np.nan,
                                "t_equal": np.nan, "p_equal": np.nan,
                                "t_welch": np.nan, "p_welch": np.nan,
                                "t_final": np.nan, "p_final": np.nan,
                                "cohens_d": np.nan
                            })
                        rows.append(row)
                    return pd.DataFrame(rows)

                with st.spinner("Computing tests…"):
                    res = compute_tests(src, group_col, chosen)

                st.markdown("#### Results (first rows)")
                st.dataframe(res.reset_index(drop=True), use_container_width=True)

                csv = res.to_csv(index=False).encode("utf-8")
                st.download_button("Download results CSV", data=csv, file_name="ttest_anova_results.csv", mime="text/csv")

                st.caption("Notes: t_final/p_final = Welch for 2 groups; for >2 groups F_statistic/F_pvalue shown. Cohen's d uses pooled-ish denom for quick effect-size estimate.")
    st.markdown('</div>', unsafe_allow_html=True)
    
# ========== STEPWISE ==========

with tab_pair:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("Pairwise Relationships (Sample, auto)")
    PAIR_NUM = [c for c in df.select_dtypes(include=[np.number]).columns if c != "target"]
    if len(PAIR_NUM) < 2:
        st.info("Need at least two numeric columns.")
    else:
        top_by_corr, _ = get_featured_vars(df, k=min(6, len(PAIR_NUM)))
        chosen = top_by_corr if top_by_corr else PAIR_NUM[:min(4, len(PAIR_NUM))]
        sample_n = min(5000, len(df))
        src = df.sample(sample_n, random_state=42) if len(df) > sample_n else df.copy()
        src = src[chosen].dropna()

        charts = []
        for i, ycol in enumerate(chosen):
            row_charts = []
            for j, xcol in enumerate(chosen):
                if i == j:
                    c = alt.Chart(src).transform_density(xcol, as_=[xcol, "density"]).mark_area(opacity=0.5).encode(
                        x=alt.X(f"{xcol}:Q", title=None), y="density:Q"
                    ).properties(height=150, width=150)
                elif i > j:
                    c = alt.Chart(src).mark_circle(size=20, opacity=0.6).encode(
                        x=alt.X(f"{xcol}:Q", title=None), y=alt.Y(f"{ycol}:Q", title=None),
                        tooltip=[xcol, ycol]
                    ).properties(height=150, width=150)
                else:
                    c = alt.Chart(pd.DataFrame({"x":[0],"y":[0]})).mark_rect(opacity=0).properties(height=150, width=150)
                row_charts.append(c)
            charts.append(alt.hconcat(*row_charts, spacing=6))
        grid = alt.vconcat(*charts, spacing=6).resolve_scale(color="independent").properties(title="Pairwise Relationships (Sample)")
        st.altair_chart(grid, use_container_width=True)
        st.caption(f"Variables shown: {', '.join(chosen)}  •  Sample: {len(src):,} rows")
    st.markdown('</div>', unsafe_allow_html=True)

# ========== Logit (own tab) ==========
with tab_logit:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("Logistic Regression — Interpret the model")
    st.caption("Target legend — 0: Charged Off, 1: Fully Paid")

    if "target" not in df.columns:
        st.info("No 'target' column found.")
    else:
        numeric_pool = [c for c in df.select_dtypes(include=[np.number]).columns if c != "target"]
        if len(numeric_pool) == 0:
            st.info("No numeric features available for logit.")
        else:
            default_pool = [c for c in ["int_rate","dti","revol_util","loan_amnt","annual_inc"] if c in numeric_pool] or numeric_pool[:8]

            with st.expander("⚙️ Model settings", expanded=False):
                C = st.slider("Regularization strength (C)", 0.01, 10.0, 1.0, 0.01)
                balance = st.checkbox("Class weight = 'balanced'", value=True)
                top_k = st.slider("Auto-select top-k features (by |coef|)", 3, min(12, len(default_pool)), 6)
                feats_override = st.multiselect("(Optional) Manually choose features", options=numeric_pool, default=default_pool)

            try:
                from sklearn.preprocessing import StandardScaler
                from sklearn.linear_model import LogisticRegression
                from sklearn.pipeline import Pipeline

                base_feats = feats_override if feats_override else default_pool
                dtrain0 = df[["target"] + base_feats].dropna().copy()
                X0 = dtrain0[base_feats].values
                y0 = pd.to_numeric(dtrain0["target"], errors="coerce").astype(int).values

                base_pipe = Pipeline([
                    ("scaler", StandardScaler()),
                    ("logit", LogisticRegression(C=C, class_weight=("balanced" if balance else None),
                                                 solver="liblinear", max_iter=400))
                ])
                base_pipe.fit(X0, y0)

                init_coefs = base_pipe.named_steps["logit"].coef_.ravel()
                order = np.argsort(-np.abs(init_coefs))
                feats = [base_feats[i] for i in order[:top_k]]

                dtrain = df[["target"] + feats].dropna().copy()
                X = dtrain[feats].values
                y = pd.to_numeric(dtrain["target"], errors="coerce").astype(int).values

                pipe = Pipeline([
                    ("scaler", StandardScaler()),
                    ("logit", LogisticRegression(C=C, class_weight=("balanced" if balance else None),
                                                 solver="liblinear", max_iter=400))
                ])
                pipe.fit(X, y)
                probs = pipe.predict_proba(X)[:, 1]
                clf = pipe.named_steps["logit"]

            except Exception as e:
                st.info("Scikit-learn is required for this tab. Add `scikit-learn` to requirements.")
                st.exception(e)
                st.markdown('</div>', unsafe_allow_html=True)
                st.stop()

            visual = st.radio("Visual type", ["Odds-ratio forest", "Probability vs one feature", "Interaction heatmap"], horizontal=True)

            if visual == "Odds-ratio forest":
                coefs = clf.coef_.ravel(); odds = np.exp(coefs)
                ci_low, ci_high = None, None
                try:
                    import statsmodels.api as sm
                    from sklearn.preprocessing import StandardScaler
                    Z = StandardScaler().fit_transform(dtrain[feats].values)
                    Z = sm.add_constant(Z)
                    sm_mod = sm.Logit(y, Z).fit(disp=False)
                    params = sm_mod.params[1:]
                    cov = sm_mod.cov_params().values[1:, 1:]
                    se = np.sqrt(np.diag(cov))
                    ci_low = np.exp(params - 1.96 * se); ci_high = np.exp(params + 1.96 * se)
                except Exception:
                    pass

                coef_df = pd.DataFrame({"feature": feats, "odds_ratio": odds}).sort_values("odds_ratio", ascending=False)
                if ci_low is not None:
                    coef_df["ci_low"] = ci_low; coef_df["ci_high"] = ci_high

                base = alt.Chart(coef_df).encode(y=alt.Y("feature:N", sort="-x", title=""))
                bars = base.mark_bar(size=10).encode(
                    x=alt.X("odds_ratio:Q", title="Odds Ratio (exp(coef))"),
                    tooltip=["feature", alt.Tooltip("odds_ratio:Q", format=".2f")]
                )
                chart = bars if "ci_low" not in coef_df else bars + base.mark_rule().encode(
                    x="ci_low:Q", x2="ci_high:Q",
                    tooltip=["feature", alt.Tooltip("odds_ratio:Q", format=".2f"),
                             alt.Tooltip("ci_low:Q", format=".2f"), alt.Tooltip("ci_high:Q", format=".2f")]
                )
                st.markdown("**Feature effects (Odds Ratios)** — > 1 increases odds of target=1; < 1 decreases.")
                st.altair_chart(chart.properties(height=360), use_container_width=True)

            elif visual == "Probability vs one feature":
                # prefer one of your EDA vars if present
                feasible = [f for f in feats if f in EDA_VARS] or feats
                one_x = feasible[0]
                plot_df = dtrain[[one_x]].copy(); plot_df["p1"] = probs
                bins = np.linspace(plot_df[one_x].min(), plot_df[one_x].max(), 31)
                plot_df["bin"] = pd.cut(plot_df[one_x], bins=bins, include_lowest=True)
                line_df = plot_df.groupby("bin", observed=False).agg(
                    x=(one_x, "mean"), p=("p1", "mean"), n=("p1", "size")
                ).dropna()
                line = alt.Chart(line_df).mark_line(point=True).encode(
                    x=alt.X("x:Q", title=one_x),
                    y=alt.Y("p:Q", title="Mean P(target=1)"),
                    size=alt.Size("n:Q", legend=None, title="Bin size"),
                    tooltip=[alt.Tooltip("x:Q", format=".2f"), alt.Tooltip("p:Q", format=".3f"), "n:Q"]
                ).properties(height=360)
                st.altair_chart(line, use_container_width=True)

            else:  # Interaction heatmap
                if len(feats) < 2:
                    st.info("Need at least two features for an interaction.")
                else:
                    f1, f2 = feats[0], feats[1]
                    tmp = dtrain[[f1, f2]].copy(); tmp["p1"] = probs
                    bx = pd.cut(tmp[f1], bins=20, include_lowest=True)
                    by = pd.cut(tmp[f2], bins=20, include_lowest=True)
                    grid = tmp.groupby([bx, by], observed=False)["p1"].mean().reset_index()
                    grid.columns = [f1, f2, "p"]
                    def mid(iv):
                        try: return (iv.left + iv.right) / 2
                        except Exception: return np.nan
                    grid["x"] = grid[f1].apply(mid); grid["y"] = grid[f2].apply(mid); grid = grid.dropna()
                    heat = alt.Chart(grid).mark_rect().encode(
                        x=alt.X("x:Q", title=f1), y=alt.Y("y:Q", title=f2),
                        color=alt.Color("p:Q", title="Mean P(target=1)"),
                        tooltip=[alt.Tooltip("x:Q", format=".2f"),
                                 alt.Tooltip("y:Q", format=".2f"),
                                 alt.Tooltip("p:Q", format=".3f")]
                    ).properties(height=420)
                    st.altair_chart(heat, use_container_width=True)

            st.caption("Exploratory interpretation only • Standardized features • No performance metrics shown.")

    st.markdown('</div>', unsafe_allow_html=True)

# -------------------- Footer --------------------
st.write("")
st.markdown(
    """
    <div style="text-align:center; color:#64748b; font-size:.9rem; padding:10px 0 0 0;">
      Camila and Altynsara ✅
    </div>
    """,
    unsafe_allow_html=True
)
