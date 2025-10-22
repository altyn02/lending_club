import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import streamlit as st
import altair as alt
import matplotlib.pyplot as plt
st.set_page_config(page_title="LendingClub Credit Modeling ", layout="wide", page_icon="💳")
alt.data_transformers.disable_max_rows()
