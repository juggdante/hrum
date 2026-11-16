"""HRUM app (Streamlit, multipage): chat with charts, question bank, dataset guide, SQL Explorer (analyst).

    make agent                 # 127.0.0.1:8501 (APP_PORT=... for another port)
"""
from __future__ import annotations

import streamlit as st

from common.config import load_config

st.set_page_config(page_title="HRUM", page_icon=":bar_chart:", layout="wide")


@st.cache_resource(show_spinner="Memuat model pencarian dokumen ...")
def _warm_up_retrieval() -> bool:
    from agent.rag.retrieve import warm_up

    return warm_up()


_warm_up_retrieval()

# A page may ask for a role change (e.g. an analyst question from the question bank). Widget state can only be
# set before the widget is created, so the request is applied here, before the sidebar radio.
if "role_request" in st.session_state:
    st.session_state["role"] = st.session_state.pop("role_request")

with st.sidebar:
    cfg = load_config("llm")
    # only providers with a client implementation (see agent/llm/base.py get_client)
    providers = [n for n, p in cfg["providers"].items() if p["kind"] in ("openai_compatible", "openai")]
    default = cfg["default"] if cfg["default"] in providers else providers[0]
    st.selectbox("Model", providers, index=providers.index(default), key="provider")
    st.radio("Role", ["chro", "analyst"], horizontal=True, key="role",
             help="CHRO: governed metrics only. Analyst: also SQL (agent tool + SQL Explorer page).")
    st.caption("Compensation data is REAL (SF, FY2013-2026). HRIS fields (turnover, tenure, gender, ...) "
               "are SYNTHETIC.")

pages = [
    st.Page("views/chat.py", title="Tanya HRUM", icon=":material/chat:", default=True),
    st.Page("views/questions.py", title="Daftar pertanyaan", icon=":material/help:"),
    st.Page("views/dataset.py", title="Panduan dataset", icon=":material/dataset:"),
]
if st.session_state.get("role") == "analyst":   # mirrors run_sql: analysts only
    pages.append(st.Page("views/sql_explorer.py", title="SQL Explorer", icon=":material/terminal:"))

st.navigation(pages).run()
