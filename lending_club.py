# app.py — Lending Club Dashboard (target-only, issue_year filter + auto-grid EDA + Logit visuals)
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
    st.markdown(f'<div class="kpi"><div class="label">Bad Rate (target==1)</div><div class="value">{bad_ratio}</div></div>', unsafe_allow_html=True)

st.write("")

# -------------------- Featured & Suitability helpers --------------------
from pandas.api.types import is_numeric_dtype

DOWNSAMPLE_MAX = 50_000  # change to 0 to disable

def get_featured_vars(df, k=6):
    """Robust ranking by |corr(target, X)| when target can be coerced numeric; else fallback."""
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

    cat_candidates = ["grade", "term", "home_ownership", "purpose", "emp_length", "verification_status"]
    top_cat = [c for c in cat_candidates if c in df.columns]
    return top_num, top_cat

def continuous_numeric_cols(df: pd.DataFrame, min_unique: int = 10, exclude: set | None = None) -> list:
    exclude = exclude or set()
    return [
        c for c in df.columns
        if c not in exclude and is_numeric_dtype(df[c]) and df[c].dropna().nunique() >= min_unique
    ]

def categorical_cols(df: pd.DataFrame, max_card: int = 30, include_target_if_cat=True) -> list:
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
        if (t.dtype.name in ("object", "category")) or (is_numeric_dtype(t) and t.dropna().nunique() <= 10):
            if "target" not in cats:
                cats = ["target"] + cats
    return list(dict.fromkeys(cats))

def numeric_cols(df: pd.DataFrame, exclude: set | None = None) -> list:
    exclude = exclude or set()
    return [c for c in df.columns if c not in exclude and is_numeric_dtype(df[c])]


# Downsample for snappy charts (EDA only)
df_eda = df.copy()

# Suitability lists
EXCLUDE = {"target"}
HIST_NUM   = continuous_numeric_cols(df_eda, min_unique=10, exclude=EXCLUDE)
DENS_NUM   = HIST_NUM[:]  # same criteria
BOX_Y_NUM  = continuous_numeric_cols(df_eda, min_unique=5, exclude=EXCLUDE)
BOX_X_CAT  = categorical_cols(df_eda, max_card=30, include_target_if_cat=True)
CORR_NUM   = numeric_cols(df_eda, exclude=set())  # allow target if numeric

# -------------------- Tabs --------------------
tab_hist, tab_box, tab_density, tab_corr, tab_pair, tab_logit = st.tabs([
    "📊 Histograms", "📦 Boxplots", "🌫️ Density (KDE)",
    "🧮 Correlation Heatmap", "🔗 Pairwise (Sample)", "🧠 Logit"
])

# ========== Histograms (auto-grid, no pickers) ==========
with tab_hist:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("Histogram — All suitable numeric variables")
    if not HIST_NUM:
        st.info("No suitable numeric columns.")
    else:
        bins = st.slider("Bins", 10, 80, 40, 5, key="hist_bins_auto")
        src = df_eda[HIST_NUM + (["target"] if "target" in df_eda.columns else [])].dropna()
        chart = (
            alt.Chart(src)
            .transform_fold(HIST_NUM, as_=["variable", "value"])
            .mark_bar(opacity=0.7)
            .encode(
                x=alt.X("value:Q", bin=alt.Bin(maxbins=bins), title=None),
                y=alt.Y("count():Q", title="Count"),
                color=alt.Color("target:N", title="target") if "target" in src.columns else alt.value(None),
                facet=alt.Facet("variable:N", columns=3, title="")
            )
            .properties(height=160)
        )
        st.altair_chart(chart, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

# ========== Boxplots (auto-grid, no pickers) ==========
with tab_box:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("Boxplot — All suitable numeric variables")
    if not BOX_Y_NUM or not BOX_X_CAT:
        st.info("Need at least one continuous numeric (Y) and one categorical (X).")
    else:
        x_col = "target" if "target" in BOX_X_CAT else BOX_X_CAT[0]
        src = df_eda[[x_col] + BOX_Y_NUM].dropna()
        melt = src.melt(id_vars=[x_col], value_vars=BOX_Y_NUM, var_name="variable", value_name="value")
        chart = (
            alt.Chart(melt)
            .mark_boxplot()
            .encode(
                x=alt.X(f"{x_col}:N", title=x_col),
                y=alt.Y("value:Q", title=None),
                color=alt.Color(f"{x_col}:N", legend=None),
                facet=alt.Facet("variable:N", columns=3, title="")
            )
            .properties(height=160)
        )
        st.altair_chart(chart, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

# ========== Density (KDE; auto-grid, no pickers) ==========
with tab_density:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("Density (KDE) — All suitable numeric variables")
    if not DENS_NUM:
        st.info("No suitable continuous numeric columns for density.")
    else:
        src = df_eda[DENS_NUM + (["target"] if "target" in df_eda.columns else [])].dropna()
        base = alt.Chart(src).transform_fold(DENS_NUM, as_=["variable", "value"])
        if "target" in src.columns:
            chart = base.transform_density(
                "value", groupby=["variable", "target"], as_=["value", "density"]
            ).mark_area(opacity=0.5).encode(
                x=alt.X("value:Q", title=None),
                y="density:Q",
                color="target:N",
                facet=alt.Facet("variable:N", columns=3, title="")
            ).properties(height=160)
        else:
            chart = base.transform_density(
                "value", groupby=["variable"], as_=["value", "density"]
            ).mark_area(opacity=0.6).encode(
                x=alt.X("value:Q", title=None),
                y="density:Q",
                facet=alt.Facet("variable:N", columns=3, title="")
            ).properties(height=160)
        st.altair_chart(chart, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

# ========== Correlation Heatmap (preset or manual) ==========
with tab_corr:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("Correlation Heatmap (numeric only)")
    num_df = df_eda.select_dtypes(include=[np.number]).copy()
    if num_df.empty or num_df.shape[1] < 2:
        st.info("Not enough numeric columns for a correlation heatmap.")
    else:
        numeric_options = list(num_df.columns)
        if "target" in numeric_options:
            numeric_options.remove("target")
            numeric_options = ["target"] + numeric_options

        preset = st.radio("Feature selection", ["Top by |corr with target|", "Manual"], horizontal=True, key="corr_mode")
        cap = st.slider("Max features to display", 3, 18, 12, key="corr_cap")

        if preset == "Top by |corr with target|":
            if "target" in num_df.columns and num_df.shape[1] > 1:
                cabs = num_df.corr(numeric_only=True)["target"].drop(labels=["target"], errors="ignore").abs()
                ranked = cabs.sort_values(ascending=False).index.tolist()
                chosen = ["target"] + ranked[: max(0, cap - 1)]
                st.caption(f"Auto-selected: {', '.join(chosen[:min(10,len(chosen))])}{'…' if len(chosen)>10 else ''}")
            else:
                chosen = numeric_options[:cap]
        else:
            chosen = st.multiselect(
                "Pick numeric features",
                options=numeric_options,
                default=(numeric_options[:cap]),
                help="Keep it ≤ 18 for readability.",
                key="corr_manual"
            )

        chosen = [c for c in chosen if c in num_df.columns]
        if len(chosen) < 2:
            st.info("Pick at least two features.")
        else:
            order_mode = st.selectbox(
                "Order axes by",
                ["Selected order", "Alphabetical", "By |corr with target| (if available)"],
                index=2 if "target" in chosen else 0,
                key="corr_order"
            )
            ordered = chosen[:]
            if order_mode == "Alphabetical":
                ordered = sorted(chosen)
            elif order_mode == "By |corr with target| (if available)" and "target" in chosen:
                cabs2 = num_df[chosen].corr(numeric_only=True)["target"].abs().sort_values(ascending=False)
                ordered = ["target"] + [c for c in cabs2.index if c != "target"]

            cmat = num_df[ordered].corr(numeric_only=True)
            corr_df = cmat.reset_index().melt("index")
            corr_df.columns = ["feature_x", "feature_y", "corr"]

            heat = alt.Chart(corr_df).mark_rect().encode(
                x=alt.X("feature_x:O", title="", sort=ordered),
                y=alt.Y("feature_y:O", title="", sort=ordered),
                color=alt.Color("corr:Q", scale=alt.Scale(scheme="blueorange", domain=[-1, 1])),
                tooltip=["feature_x", "feature_y", alt.Tooltip("corr:Q", format=".2f")]
            ).properties(height=420)

            show_labels = st.checkbox("Show correlation values", value=False, key="corr_labels")
            if show_labels:
                labels = alt.Chart(corr_df).mark_text(baseline="middle").encode(
                    x=alt.X("feature_x:O", sort=ordered),
                    y=alt.Y("feature_y:O", sort=ordered),
                    text=alt.Text("corr:Q", format=".2f"),
                    color=alt.condition("datum.corr > 0.5 || datum.corr < -0.5", alt.value("white"), alt.value("black"))
                )
                st.altair_chart(heat + labels, use_container_width=True)
            else:
                st.altair_chart(heat, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

# ========== Pairwise (scatter-matrix style; auto) ==========
with tab_pair:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("Pairwise Relationships (Sample, auto)")
    PAIR_NUM = [c for c in df.select_dtypes(include=[np.number]).columns if c != "target"]
    if len(PAIR_NUM) < 2:
        st.info("Need at least two numeric columns.")
    else:
        # choose top variables by |corr(target, X)| if possible
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
                    c = (
                        alt.Chart(src)
                        .transform_density(xcol, as_=[xcol, "density"])
                        .mark_area(opacity=0.5)
                        .encode(x=alt.X(f"{xcol}:Q", title=None), y="density:Q")
                        .properties(height=150, width=150)
                    )
                elif i > j:
                    c = (
                        alt.Chart(src)
                        .mark_circle(size=20, opacity=0.6)
                        .encode(
                            x=alt.X(f"{xcol}:Q", title=None),
                            y=alt.Y(f"{ycol}:Q", title=None),
                            tooltip=[xcol, ycol]
                        )
                        .properties(height=150, width=150)
                    )
                else:
                    c = alt.Chart(pd.DataFrame({"x":[0],"y":[0]})).mark_rect(opacity=0).properties(height=150, width=150)
                row_charts.append(c)
            charts.append(alt.hconcat(*row_charts, spacing=6))
        grid = alt.vconcat(*charts, spacing=6).resolve_scale(color="independent").properties(title="Pairwise Relationships (Sample)")
        st.altair_chart(grid, use_container_width=True)
        st.caption(f"Variables shown: {', '.join(chosen)}  •  Sample: {len(src):,} rows")
    st.markdown('</div>', unsafe_allow_html=True)

# ========== Logit (focused UI, no performance evaluation) ==========
st.markdown('<div class="card">', unsafe_allow_html=True)
st.subheader("Logistic Regression — Interpret the model")

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
            coefs = clf.coef_.ravel()
            odds = np.exp(coefs)
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
                ci_low = np.exp(params - 1.96 * se)
                ci_high = np.exp(params + 1.96 * se)
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
            # prefer continuous variables among selected features
            feasible = [f for f in feats if f in HIST_NUM] or feats
            one_x = feasible[0]
            plot_df = dtrain[[one_x]].copy()
            plot_df["p1"] = probs
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

        else:
            if len(feats) < 2:
                st.info("Need at least two features for an interaction.")
            else:
                f1, f2 = feats[0], feats[1]
                tmp = dtrain[[f1, f2]].copy()
                tmp["p1"] = probs
                bx = pd.cut(tmp[f1], bins=20, include_lowest=True)
                by = pd.cut(tmp[f2], bins=20, include_lowest=True)
                grid = tmp.groupby([bx, by], observed=False)["p1"].mean().reset_index()
                grid.columns = [f1, f2, "p"]

                def mid(iv):
                    try: return (iv.left + iv.right) / 2
                    except Exception: return np.nan
                grid["x"] = grid[f1].apply(mid); grid["y"] = grid[f2].apply(mid); grid = grid.dropna()

                heat = alt.Chart(grid).mark_rect().encode(
                    x=alt.X("x:Q", title=f1),
                    y=alt.Y("y:Q", title=f2),
                    color=alt.Color("p:Q", title="Mean P(target=1)"),
                    tooltip=[alt.Tooltip("x:Q", format=".2f"), alt.Tooltip("y:Q", format=".2f"), alt.Tooltip("p:Q", format=".3f")]
                ).properties(height=420)
                st.altair_chart(heat, use_container_width=True)

        st.caption("Exploratory interpretation only • Standardized features • No performance metrics shown.")

st.markdown('</div>', unsafe_allow_html=True)

# -------------------- Footer --------------------
st.write("")
st.markdown(
    """
    <div style="text-align:center; color:#64748b; font-size:.9rem; padding:10px 0 0 0;">
      Lending Club Camila and Altynsara ✅
    </div>
    """,
    unsafe_allow_html=True
)
