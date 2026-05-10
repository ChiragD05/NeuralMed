import pandas as pd
import streamlit as st

from utils.ui import load_css, page_header
from services.db import fetch_my_rows

TABLES = [
    ("Symptom Logs", "symptom_logs"),
    ("Medical Images", "medical_images"),
    ("Report Summaries", "report_summaries"),
    ("Prescriptions", "prescriptions"),
    ("Risk Predictions", "risk_predictions"),
    ("Chat Messages", "chat_messages"),
    ("Voice Sessions", "voice_sessions"),
]

def _show_table(title, table_name):
    result = fetch_my_rows(table_name, limit=10)
    rows = result.data if hasattr(result, "data") else []

    st.subheader(title)

    if rows:
        df = pd.DataFrame(rows)
        st.dataframe(df, use_container_width=True)
    else:
        st.info("No saved records yet.")

def render():
    load_css()
    page_header("📚 My History", "Your saved AI interactions")

    st.caption("This page shows only records saved by the currently logged-in user.")

    tabs = st.tabs([x[0] for x in TABLES])
    for tab, (_, table_name) in zip(tabs, TABLES):
        with tab:
            _show_table(table_name.replace("_", " ").title(), table_name)