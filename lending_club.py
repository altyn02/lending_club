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
# -------------------- Tabs (merged Hist+Box into one) --------------------
tab_data, tab_dist, tab_corr, tab_ttest, tab_cv, tab_logit = st.tabs([
    "🧭Data Exploration", "📈 Data Visualization", "🧮 Correlation Heatmap",
    "📏 t-Tests", "🏁 Performance Evaluation ", "🧠 Logit"
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
    
# ========== Distributions (merged Histograms + Boxplots + Line) ==========
with tab_dist:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("Distributions — Histograms, Boxplots & Line")
    st.caption("🎯 Target legend: 0 = Charged Off, 1 = Fully Paid")

    if not EDA_VARS:
        st.info("No suitable numeric columns from the requested list.")
    else:
        # --- Controls row
        c1, c2, c3, c4 = st.columns([1, 1, 1, 2])
        with c1:
            show_hist = st.checkbox("Histograms", value=True, key="dist_show_hist")
        with c2:
            show_box  = st.checkbox("Boxplots",  value=True, key="dist_show_box")
        with c3:
            show_line = st.checkbox("Line",      value=True, key="dist_show_line")
        with c4:
            bins = st.slider("Histogram bins", 10, 80, 40, 5, key="dist_bins")

        # Base data
        HIST_BOX_VARS = [v for v in EDA_VARS if v != "int_rate"]  # exclude int_rate
        src_all = df_eda[EDA_VARS + (["target"] if "target" in df_eda.columns else [])].dropna()

        # -------------------- Line chart (time trend) --------------------
        if show_line:
            time_candidates = [c for c in ["issue_year", "issue_d"] if c in df.columns]
            time_col = st.selectbox(
                "Time column for line chart",
                options=time_candidates if time_candidates else [],
                index=0 if time_candidates else None,
                help="Uses issue_year if available; otherwise choose a time-like column."
            )

            y_var = st.selectbox(
                "Y variable",
                options=EDA_VARS,  # full list; int_rate allowed for Line
                index=(EDA_VARS.index("loan_amnt") if "loan_amnt" in EDA_VARS else 0),
                key="line_y_var"
            )
            agg_choice = st.selectbox("Aggregation", ["Mean", "Median", "Count"], index=0, key="line_agg")

            if not time_col:
                st.info("No time column available (e.g., issue_year).")
            else:
                df_line = df[[time_col, y_var] + (["target"] if "target" in df.columns else [])].copy()

                if time_col == "issue_year":
                    df_line["__time__"] = pd.to_numeric(df_line[time_col], errors="coerce")
                else:
                    dt = pd.to_datetime(df_line[time_col], errors="coerce")
                    df_line["__time__"] = dt.dt.year

                df_line = df_line.dropna(subset=["__time__"])

                if agg_choice == "Mean":
                    agg_func, y_enc_title = "mean", f"Mean {y_var}"
                elif agg_choice == "Median":
                    agg_func, y_enc_title = "median", f"Median {y_var}"
                else:
                    agg_func, y_enc_title = "count", f"Count ({y_var} non-null)"

                if "target" in df_line.columns:
                    g = df_line.groupby(["__time__", "target"], observed=False)
                else:
                    g = df_line.groupby(["__time__"], observed=False)

                plot_df = (g[y_var].count().reset_index(name="y")
                           if agg_choice == "Count"
                           else g[y_var].agg(agg_func).reset_index(name="y"))

                enc_color = alt.Color("target:N", title="target") if "target" in plot_df.columns else alt.value(None)

                line = (
                    alt.Chart(plot_df)
                    .mark_line(point=True)
                    .encode(
                        x=alt.X("__time__:O", title=("Issue Year" if time_col == "issue_year" else time_col)),
                        y=alt.Y("y:Q", title=y_enc_title),
                        color=enc_color,
                        tooltip=[alt.Tooltip("__time__:O", title="Time"),
                                 alt.Tooltip("y:Q", title=y_enc_title, format=".2f")]
                                + ([alt.Tooltip("target:N", title="target")] if "target" in plot_df.columns else [])
                    )
                    .properties(height=300, title=f"{agg_choice} {y_var} by {time_col}")
                )
                st.altair_chart(line, use_container_width=True)

        # -------------------- Histogram (single variable, EXCLUDING int_rate) --------------------
        if show_hist:
            if not HIST_BOX_VARS:
                st.info("No variables available for histograms.")
            else:
                h_var = st.selectbox(
                    "Histogram variable",
                    options=HIST_BOX_VARS,
                    index=(HIST_BOX_VARS.index("loan_amnt") if "loan_amnt" in HIST_BOX_VARS else 0),
                    key="hist_var_single",
                )
                h_src = df_eda[[h_var] + (["target"] if "target" in df_eda.columns else [])].dropna()

                if "target" in h_src.columns:
                    hist = (
                        alt.Chart(h_src)
                        .mark_bar(opacity=0.6)
                        .encode(
                            x=alt.X(f"{h_var}:Q", bin=alt.Bin(maxbins=bins), title=h_var),
                            y=alt.Y("count():Q", title="Count"),
                            color=alt.Color("target:N", title="target"),
                            tooltip=[alt.Tooltip(f"{h_var}:Q", title=h_var), "count()", alt.Tooltip("target:N", title="target")]
                        )
                        .properties(height=280, title=f"Histogram — {h_var}")
                    )
                else:
                    hist = (
                        alt.Chart(h_src)
                        .mark_bar(opacity=0.8)
                        .encode(
                            x=alt.X(f"{h_var}:Q", bin=alt.Bin(maxbins=bins), title=h_var),
                            y=alt.Y("count():Q", title="Count"),
                            tooltip=[alt.Tooltip(f"{h_var}:Q", title=h_var), "count()"]
                        )
                        .properties(height=280, title=f"Histogram — {h_var}")
                    )
                st.altair_chart(hist, use_container_width=True)

        # -------------------- Boxplot (single variable, EXCLUDING int_rate) --------------------
        if show_box:
            if "target" not in df_eda.columns:
                st.info("Boxplots require 'target' to group by.")
            elif not HIST_BOX_VARS:
                st.info("No variables available for boxplots.")
            else:
                b_var = st.selectbox(
                    "Boxplot variable",
                    options=HIST_BOX_VARS,
                    index=(HIST_BOX_VARS.index("annual_inc") if "annual_inc" in HIST_BOX_VARS else 0),
                    key="box_var_single",
                )
                b_src = df_eda[["target", b_var]].dropna()

                box = (
                    alt.Chart(b_src)
                    .mark_boxplot()
                    .encode(
                        x=alt.X("target:N", title="target"),
                        y=alt.Y(f"{b_var}:Q", title=b_var),
                        color=alt.Color("target:N", legend=None),
                        tooltip=[alt.Tooltip(f"{b_var}:Q", title=b_var), alt.Tooltip("target:N", title="target")]
                    )
                    .properties(height=280, title=f"Boxplot — {b_var} by target")
                )
                st.altair_chart(box, use_container_width=True)

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
    st.subheader("Welch’s t-tests (0 = Charged Off, 1 = Fully Paid)")

    if "target" not in df.columns:
        st.info("No 'target' column found.")
    else:
        import numpy as np
        import pandas as pd
        from scipy import stats

        tnum = pd.to_numeric(df["target"], errors="coerce")
        mask_valid = tnum.isin([0, 1])
        if mask_valid.sum() < 2 or tnum[mask_valid].nunique() < 2:
            st.info("Both target groups must be present to run t-tests.")
        else:
            VARS = [c for c in EDA_VARS if c in df.columns and pd.api.types.is_numeric_dtype(df[c])]
            if not VARS:
                st.info("No numeric variables available for t-tests.")
            else:
                apply_fdr = st.checkbox("Apply FDR correction (Benjamini–Hochberg)", value=True)
                rows = []
                for col in VARS:
                    s = pd.to_numeric(df[col], errors="coerce")
                    d = pd.DataFrame({"y": s, "t": tnum}).dropna()
                    g0 = d.loc[d["t"] == 0, "y"].values
                    g1 = d.loc[d["t"] == 1, "y"].values
                    if len(g0) < 2 or len(g1) < 2:
                        continue

                    m0, m1 = np.mean(g0), np.mean(g1)
                    s0, s1 = np.std(g0, ddof=1), np.std(g1, ddof=1)
                    diff = m1 - m0
                    tstat, pval = stats.ttest_ind(g1, g0, equal_var=False)

                    v0, v1 = s0**2, s1**2
                    se2 = v0/len(g0) + v1/len(g1)
                    df_welch = (se2**2) / (((v0/len(g0))**2)/(len(g0)-1) + ((v1/len(g1))**2)/(len(g1)-1))
                    tcrit = stats.t.ppf(0.975, df_welch)
                    ci_low = diff - tcrit*np.sqrt(se2)
                    ci_high = diff + tcrit*np.sqrt(se2)

                    sp2 = (((len(g0)-1)*v0)+((len(g1)-1)*v1)) / (len(g0)+len(g1)-2)
                    sp = np.sqrt(sp2)
                    d_cohen = diff / sp if np.isfinite(sp) and sp > 0 else np.nan
                    J = 1 - (3 / (4*(len(g0)+len(g1)) - 9))
                    g_hedges = d_cohen * J

                    rows.append({
                        "variable": col,
                        "n_0": len(g0), "n_1": len(g1),
                        "mean_0": m0, "mean_1": m1,
                        "std_0": s0, "std_1": s1,
                        "diff_(1-0)": diff,
                        "t": tstat, "df": df_welch, "p_value": pval,
                        "cohen_d": d_cohen, "hedges_g": g_hedges,
                        "ci_low": ci_low, "ci_high": ci_high
                    })

                if not rows:
                    st.info("No valid data to compute t-tests.")
                else:
                    res = pd.DataFrame(rows)
                    if apply_fdr:
                        p = res["p_value"].values
                        m = len(p)
                        order = np.argsort(p)
                        ranks = np.empty_like(order)
                        ranks[order] = np.arange(1, m+1)
                        q = p * m / ranks
                        q_adj = np.minimum.accumulate(q[np.argsort(order)][::-1])[::-1]
                        res["q_value"] = q_adj
                        res["significant"] = res["q_value"] < 0.05
                    else:
                        res["significant"] = res["p_value"] < 0.05

                    st.dataframe(res, use_container_width=True)
                    st.caption("Welch’s t-test (unequal variances). CI = 95% for mean difference (1−0). Effect size = Cohen’s d (Hedges’ g corrected).")

    st.markdown('</div>', unsafe_allow_html=True)


# ========== Performance  ==========
   
with tab_cv:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("5-Fold Cross-Validation Performance (F1-optimized)")
    st.caption("Target legend — 0: Charged Off, 1: Fully Paid")

    if "target" not in df.columns:
        st.info("No 'target' column found.")
    else:
        # --- Independent model settings for CV (decoupled from Logit tab)
        numeric_pool = [c for c in df.select_dtypes(include=[np.number]).columns if c != "target"]
        if len(numeric_pool) == 0:
            st.info("No numeric features available for CV.")
        else:
            default_pool = [c for c in ["int_rate","dti","revol_util","loan_amnt","annual_inc"] if c in numeric_pool] or numeric_pool[:8]

            with st.expander("⚙️ CV settings (features & regularization)", expanded=False):
                C_cv = st.slider("Regularization strength (C)", 0.01, 10.0, 1.0, 0.01, key="cv_C")
                balance_cv = st.checkbox("Class weight = 'balanced'", value=True, key="cv_bal")
                top_k_cv = st.slider("Auto-select top-k features (by |coef|)", 3, min(12, len(default_pool)), 6, key="cv_topk")
                feats_override_cv = st.multiselect(
                    "(Optional) Manually choose features",
                    options=numeric_pool,
                    default=default_pool,
                    key="cv_feats_override"
                )

            # --- Build a feature set for CV: quick prefit to rank by |coef|
            from sklearn.preprocessing import StandardScaler
            from sklearn.linear_model import LogisticRegression
            from sklearn.pipeline import Pipeline
            try:
                base_feats_cv = feats_override_cv if feats_override_cv else default_pool
                dtrain0_cv = df[["target"] + base_feats_cv].dropna().copy()
                if dtrain0_cv.empty:
                    st.info("Not enough non-missing rows for the selected features.")
                    st.markdown('</div>', unsafe_allow_html=True)
                    st.stop()

                X0_cv = dtrain0_cv[base_feats_cv].values
                y0_cv = pd.to_numeric(dtrain0_cv["target"], errors="coerce").astype(int).values

                base_pipe_cv = Pipeline([
                    ("scaler", StandardScaler()),
                    ("logit", LogisticRegression(
                        C=C_cv, class_weight=("balanced" if balance_cv else None),
                        solver="liblinear", max_iter=400))
                ])
                base_pipe_cv.fit(X0_cv, y0_cv)

                init_coefs_cv = base_pipe_cv.named_steps["logit"].coef_.ravel()
                order_cv = np.argsort(-np.abs(init_coefs_cv))
                feats_cv = [base_feats_cv[i] for i in order_cv[:top_k_cv]]
            except Exception as e:
                st.info("Scikit-learn is required for this tab. Add `scikit-learn` to requirements.")
                st.exception(e)
                st.markdown('</div>', unsafe_allow_html=True)
                st.stop()

            st.write("**Features used for CV:**", ", ".join(feats_cv))
            run_cv = st.button("Run 5-fold CV", key="run_cv_button")

            if run_cv:
                from sklearn.model_selection import StratifiedKFold
                from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix, roc_auc_score

                data = df[["target"] + feats_cv].dropna().copy()
                X_all = data[feats_cv].values
                y_all = pd.to_numeric(data["target"], errors="coerce").astype(int).values

                def best_threshold_for_f1(y_true, probs):
                    thr_grid = np.linspace(0.05, 0.95, 181)
                    best_thr, best_f1 = 0.5, -1.0
                    for thr in thr_grid:
                        y_hat = (probs >= thr).astype(int)
                        f1 = f1_score(y_true, y_hat, average="binary", zero_division=0)
                        if f1 > best_f1:
                            best_f1, best_thr = f1, thr
                    return best_thr, best_f1

                skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
                rows, cms, reports = [], [], []

                for fold_id, (tr_idx, va_idx) in enumerate(skf.split(X_all, y_all), start=1):
                    X_tr, X_va = X_all[tr_idx], X_all[va_idx]
                    y_tr, y_va = y_all[tr_idx], y_all[va_idx]

                    scaler = StandardScaler().fit(X_tr)
                    X_tr_s = scaler.transform(X_tr)
                    X_va_s = scaler.transform(X_va)

                    logit = LogisticRegression(
                        C=C_cv,
                        class_weight=("balanced" if balance_cv else None),
                        max_iter=1000,
                        solver="liblinear",
                        random_state=42
                    )
                    logit.fit(X_tr_s, y_tr)

                    p_tr = logit.predict_proba(X_tr_s)[:, 1]
                    p_va = logit.predict_proba(X_va_s)[:, 1]

                    best_thr, _ = best_threshold_for_f1(y_va, p_va)
                    y_tr_hat = (p_tr >= best_thr).astype(int)
                    y_va_hat = (p_va >= best_thr).astype(int)

                    tr_acc = accuracy_score(y_tr, y_tr_hat)
                    va_acc = accuracy_score(y_va, y_va_hat)
                    va_f1 = f1_score(y_va, y_va_hat, average="binary", zero_division=0)
                    va_auc = roc_auc_score(y_va, p_va)

                    cm = confusion_matrix(y_va, y_va_hat, labels=[0, 1])
                    rep = classification_report(y_va, y_va_hat, digits=3, zero_division=0)
                    cms.append(cm); reports.append((fold_id, rep))

                    rows.append({
                        "fold": fold_id,
                        "best_thr": round(float(best_thr), 3),
                        "train_acc": round(tr_acc, 4),
                        "val_acc": round(va_acc, 4),
                        "val_f1": round(va_f1, 4),
                        "val_auc": round(va_auc, 4),
                        "support_0": int((y_va == 0).sum()),
                        "support_1": int((y_va == 1).sum()),
                    })

                results_df = pd.DataFrame(rows)
                st.subheader("Per-Fold Results")
                st.dataframe(results_df, use_container_width=True)

                avg = results_df.mean(numeric_only=True)
                st.subheader("Averages (5-Fold)")
                st.write(
                    f"**Mean Train Acc:** {avg['train_acc']:.4f} | "
                    f"**Mean Val Acc:** {avg['val_acc']:.4f} | "
                    f"**Mean Val F1:** {avg['val_f1']:.4f} | "
                    f"**Mean Val AUC:** {avg['val_auc']:.4f} | "
                    f"**Mean Best Thr:** {avg['best_thr']:.3f}"
                )

                st.subheader("Confusion Matrices")
                total_cm = np.zeros((2, 2), dtype=int)
                for i, cm in enumerate(cms, start=1):
                    total_cm += cm
                    with st.expander(f"Fold {i} confusion matrix & report"):
                        cm_df = pd.DataFrame(cm, index=["True 0","True 1"], columns=["Pred 0","Pred 1"])
                        st.dataframe(cm_df, use_container_width=True)
                        st.code(reports[i-1][1], language="text")

                st.subheader("Aggregated Confusion Matrix")
                cm_df_total = pd.DataFrame(total_cm, index=["True 0","True 1"], columns=["Pred 0","Pred 1"])
                st.dataframe(cm_df_total, use_container_width=True)
                st.caption("Threshold is optimized per fold to maximize F1 for the positive class (1 = Fully Paid).")

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
