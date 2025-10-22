# app.py — Lending Club Design-First Dashboard
# minimal, aesthetic EDA with your merged dataset (URL or upload)

import os, io, tempfile, requests
import numpy as np
import pandas as pd
import streamlit as st
import altair as alt
from datetime import datetime

# -------------------- Page & Theme --------------------
st.set_page_config(page_title="Lending Club Dashboard", page_icon="💳", layout="wide")
alt.data_transformers.disable_max_rows()
pd.set_option("display.max_columns", 200)

# -------------------- CSS (Design) --------------------
CSS = """
<style>
/* global tweaks */
html, body, [class*="css"] { font-family: 'Inter', system-ui, -apple-system, Segoe UI, Roboto, 'Helvetica Neue', Arial, 'Apple Color Emoji','Segoe UI Emoji'; }
section.main > div { padding-top: 1rem; }

/* gradient hero */
.hero {
  background: linear-gradient(135deg, #0ea5e9 0%, #8b5cf6 100%);
  color: white; border-radius: 20px; padding: 28px 28px;
  box-shadow: 0 8px 30px rgba(27,31,35,.15);
}

/* card */
.card {
  background: white; border-radius: 16px; padding: 18px 18px;
  box-shadow: 0 10px 30px rgba(0,0,0,.06); border: 1px solid rgba(0,0,0,.04);
}

/* metric cards */
.kpi { border-radius: 16px; padding: 14px 16px; background: #f8fafc; border: 1px solid #e5e7eb; }
.kpi .label { font-size: 0.92rem; color: #475569; }
