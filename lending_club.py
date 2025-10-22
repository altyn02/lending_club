# app.py — Lending Club Dashboard (target-only, issue_year filter + Logit visuals)
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
SUBTITLE = "Explore distributions, relationships, and correlations in lending club"
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

# -------------------- Sidebar Filters (issue_year slider) --------------------
with st.sidebar:
    st.subheader("Filters")

    year_range = None
    if "issue_year" in df_full.columns and df_full["issue_year"].notna().any():
        years = pd.to_numeric(df_full["issue_year"], errors="coerce").dropna().astype(int)
        min_year, max_year = int(years.min()), int(years.max())
        year_range = st.slider(
            "Filter by Issue Year",
            min_value=min_year, max_value=max_year,
            value=(min_year, max_year)
        )
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
    bad = (df["target"].astype(int) == 1).mean()
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
tab_hist, tab_box, tab_density, tab_corr, tab_logit = st.tabs([
    "📊 Histograms", "📦 Boxplots", "🌫️ Density (KDE)", "🧮 Correlation Heatmap", "🧠 Logit"
])

# Prepare numeric column list once
num_cols = [c for c in df.select_dtypes(include=[np.number]).columns if c != "target"]

def default_index(cols, preferred):
    try:
        return cols.index(preferred)
    except Exception:
        return 0 if cols else 0

# ========== Histograms ==========
with tab_hist:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("Histogram")
    if len(num_cols) == 0:
        st.info("No numeric columns available.")
    else:
        col = st.selectbox("Numeric column", options=num_cols, index=default_index(num_cols, "int_rate"))
        bins = st.slider("Bins", 10, 80, 40, 5)
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
        y_col = st.selectbox("Y (numeric)", options=num_cols, index=default_index(num_cols, "annual_inc"), key="box_y")
        cat_options = []
        if "target" in df.columns:
            cat_options.append("target")
        cat_options += [c for c in df.columns if df[c].dtype == "object" or df[c].dtype.name == "category"]
        cat_options = list(dict.fromkeys(cat_options))
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
        dens_col = st.selectbox("Numeric column", options=num_cols, index=default_index(num_cols, "dti"), key="dens")
        if "target" in df.columns:
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
    num_df = df.select_dtypes(include=[np.number]).copy()
    if num_df.empty:
        st.info("No numeric columns to correlate.")
    else:
        default_features = list(num_df.columns)
        if "target" in num_df.columns and len(num_df.columns) > 1:
            corr_to_target = num_df.corr(numeric_only=True)["target"].abs().sort_values(ascending=False)
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

# ========== Logit (focused UI, no performance evaluation) ==========
st.markdown('<div class="card">', unsafe_allow_html=True)
st.subheader("Logistic Regression — Interpret the model")

if "target" not in df.columns:
    st.info("No 'target' column found.")
else:
    # ---------- 1) Choose (or auto-pick) features ----------
    numeric_pool = [c for c in df.select_dtypes(include=[np.number]).columns if c != "target"]
    if len(numeric_pool) == 0:
        st.info("No numeric features available for logit.")
    else:
        # sensible defaults
        default_pool = [c for c in ["int_rate","dti","revol_util","loan_amnt","annual_inc"] if c in numeric_pool]
        if not default_pool:
            default_pool = numeric_pool[:8]

        with st.expander("⚙️ Model settings", expanded=False):
            C = st.slider("Regularization strength (C)", 0.01, 10.0, 1.0, 0.01)
            balance = st.checkbox("Class weight = 'balanced'", value=True)
            top_k = st.slider("Auto-select top-k features (by |coef|)", 3, min(12, len(default_pool)), 6)
            # optional manual override
            feats_override = st.multiselect("(Optional) Manually choose features", options=numeric_pool, default=default_pool)

        # Fit once on a candidate set (we’ll re-rank and then refit on top-k)
        try:
            from sklearn.preprocessing import StandardScaler
            from sklearn.linear_model import LogisticRegression
            from sklearn.pipeline import Pipeline

            base_feats = feats_override if feats_override else default_pool
            dtrain0 = df[["target"] + base_feats].dropna().copy()
            X0 = dtrain0[base_feats].values
            y0 = dtrain0["target"].astype(int).values

            base_pipe = Pipeline([
                ("scaler", StandardScaler()),
                ("logit", LogisticRegression(C=C, class_weight=("balanced" if balance else None),
                                             solver="liblinear", max_iter=400))
            ])
            base_pipe.fit(X0, y0)

            # rank by |coef| then keep top_k
            init_coefs = base_pipe.named_steps["logit"].coef_.ravel()
            order = np.argsort(-np.abs(init_coefs))
            feats = [base_feats[i] for i in order[:top_k]]

            # final fit on top-k
            dtrain = df[["target"] + feats].dropna().copy()
            X = dtrain[feats].values
            y = dtrain["target"].astype(int).values

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

        # ---------- 2) Visual type (simple radio) ----------
        visual = st.radio(
            "Visual type",
            ["Odds-ratio forest", "Probability vs one feature", "Interaction heatmap"],
            horizontal=True
        )

        # ---------- 3a) Odds-ratio forest (with CI if statsmodels available) ----------
        if visual == "Odds-ratio forest":
            coefs = clf.coef_.ravel()
            odds = np.exp(coefs)

            # Try to compute CIs via statsmodels (optional)
            ci_low, ci_high = None, None
            try:
                import statsmodels.api as sm
                # refit with statsmodels on standardized X so signs match
                from sklearn.preprocessing import StandardScaler
                Z = StandardScaler().fit_transform(dtrain[feats].values)
                Z = sm.add_constant(Z)
                sm_mod = sm.Logit(y, Z).fit(disp=False)
                params = sm_mod.params[1:]  # drop intercept
                cov = sm_mod.cov_params().values[1:, 1:]
                se = np.sqrt(np.diag(cov))
                ci_low = np.exp(params - 1.96 * se)
                ci_high = np.exp(params + 1.96 * se)
            except Exception:
                pass  # show point estimates only

            coef_df = pd.DataFrame({
                "feature": feats,
                "odds_ratio": odds,
            }).sort_values("odds_ratio", ascending=False)

            if ci_low is not None:
                coef_df["ci_low"] = ci_low
                coef_df["ci_high"] = ci_high

            base = alt.Chart(coef_df).encode(
                y=alt.Y("feature:N", sort="-x", title="")
            )
            bars = base.mark_bar(size=10).encode(
                x=alt.X("odds_ratio:Q", title="Odds Ratio (exp(coef))"),
                tooltip=["feature", alt.Tooltip("odds_ratio:Q", format=".2f")]
            )
            if "ci_low" in coef_df:
                rules = base.mark_rule().encode(
                    x="ci_low:Q", x2="ci_high:Q",
                    tooltip=[
                        "feature",
                        alt.Tooltip("odds_ratio:Q", format=".2f"),
                        alt.Tooltip("ci_low:Q", format=".2f"),
                        alt.Tooltip("ci_high:Q", format=".2f"),
                    ]
                )
                chart = bars + rules
            else:
                chart = bars

            st.markdown("**Feature effects (Odds Ratios)** — > 1 increases odds of target=1; < 1 decreases.")
            st.altair_chart(chart.properties(height=360), use_container_width=True)

        # ---------- 3b) Probability vs one feature (binned line) ----------
        elif visual == "Probability vs one feature":
            one_x = st.selectbox("Feature", options=feats, index=0)
            plot_df = dtrain[[one_x]].copy()
            plot_df["p1"] = probs

            bins = np.linspace(plot_df[one_x].min(), plot_df[one_x].max(), 31)
            plot_df["bin"] = pd.cut(plot_df[one_x], bins=bins, include_lowest=True)
            line_df = plot_df.groupby("bin", observed=False).agg(
                x=(one_x, "mean"),
                p=("p1", "mean"),
                n=("p1", "size")
            ).dropna()

            line = alt.Chart(line_df).mark_line(point=True).encode(
                x=alt.X("x:Q", title=one_x),
                y=alt.Y("p:Q", title="Mean P(target=1)"),
                size=alt.Size("n:Q", legend=None, title="Bin size"),
                tooltip=[alt.Tooltip("x:Q", format=".2f"), alt.Tooltip("p:Q", format=".3f"), "n:Q"]
            ).properties(height=360)
            st.altair_chart(line, use_container_width=True)

        # ---------- 3c) Interaction heatmap (two features) ----------
        else:  # "Interaction heatmap"
            if len(feats) < 2:
                st.info("Need at least two features for an interaction.")
            else:
                c1, c2 = st.columns(2)
                with c1:
                    f1 = st.selectbox("Feature X", options=feats, index=0, key="ix1")
                with c2:
                    f2 = st.selectbox("Feature Y", options=[f for f in feats if f != f1],
                                      index=0 if len(feats) < 3 else 1, key="ix2")

                # coarse grid → mean prob in each cell
                tmp = dtrain[[f1, f2]].copy()
                tmp["p1"] = probs
                bx = pd.cut(tmp[f1], bins=20, include_lowest=True)
                by = pd.cut(tmp[f2], bins=20, include_lowest=True)
                grid = tmp.groupby([bx, by], observed=False)["p1"].mean().reset_index()
                grid.columns = [f1, f2, "p"]

                # use interval midpoints for axes
                def mid(iv):
                    try:
                        return (iv.left + iv.right) / 2
                    except Exception:
                        return np.nan
                grid["x"] = grid[f1].apply(mid)
                grid["y"] = grid[f2].apply(mid)
                grid = grid.dropna()

                heat = alt.Chart(grid).mark_rect().encode(
                    x=alt.X("x:Q", title=f1),
                    y=alt.Y("y:Q", title=f2),
                    color=alt.Color("p:Q", title="Mean P(target=1)"),
                    tooltip=[alt.Tooltip("x:Q", format=".2f"),
                             alt.Tooltip("y:Q", format=".2f"),
                             alt.Tooltip("p:Q", format=".3f")]
                ).properties(height=420)
                st.altair_chart(heat, use_container_width=True)

        # small footnote
        st.caption("Exploratory interpretation only • Standardized features • No performance metrics shown.")

st.markdown('</div>', unsafe_allow_html=True)


# -------------------- Footer --------------------
st.write("")
st.markdown(
    """
    <div style="text-align:center; color:#64748b; font-size:.9rem; padding:10px 0 0 0;">
      Interactive Streamlit dashboard • Year slider + filters • Logit interpretation✅
    </div>
    """,
    unsafe_allow_html=True
)
