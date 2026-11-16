"""Chat page: answers with charts built from the cited evidence, then the evidence itself (rule 8)."""
from __future__ import annotations

import uuid

import pandas as pd
import streamlit as st

from agent.app.charts import charts_for_evidence
from agent.app.data import catalog, is_dark
from agent.core.loop import answer_question
from agent.llm.base import get_client

st.title("HRUM: tanya data HR")
st.caption("Setiap angka berasal dari tool yang dikutip; grafik digambar dari data bukti yang sama. "
           "Lihat **Daftar pertanyaan** untuk ide, **Panduan dataset** untuk isi datanya.")

if "history" not in st.session_state:
    st.session_state["history"] = []


def render_charts(result: dict, turn: int) -> None:
    for i, spec in enumerate(charts_for_evidence(result["evidence"], catalog(), dark=is_dark())):
        if spec.kind == "kpi":
            cols = st.columns(min(len(spec.kpis), 4))
            for j, (label, value) in enumerate(spec.kpis):
                cols[j % len(cols)].metric(label, value)
        else:
            st.plotly_chart(spec.figure, theme="streamlit", key=f"chart-{turn}-{i}")
        if spec.caption:
            st.caption(spec.caption)


def show(result: dict, turn: int) -> None:
    badge = {"answered": ":green[answered]", "abstain": ":orange[abstain]", "refused": ":red[refused]"}
    st.markdown(f"{badge.get(result['status'], result['status'])}  \n{result['answer']}")
    if result["status"] == "answered":
        render_charts(result, turn)
    if result["caveats"]:
        st.info("\n".join(f"- {c}" for c in result["caveats"]))
    with st.expander(f"Evidence ({len(result['evidence'])}) & grounding"):
        for e in result["evidence"]:
            st.markdown(f"**{e['id']}** `{e['tool']}` {'(cited)' if e['cited'] else ''}")
            if e.get("definitions"):
                for m, d in e["definitions"].items():
                    st.markdown(f"- `{m}`: {d}")
            if e.get("sql"):
                st.code(e["sql"], language="sql")
            if e.get("rows"):
                st.dataframe(pd.DataFrame(e["rows"]).drop(columns=["group_size"], errors="ignore"), hide_index=True)
            if e.get("result") is not None:
                st.write(e["result"])
        g = result["grounding"]
        st.caption(f"grounding ok={g['ok']} claims={g['claims_ok']}/{g['claims_total']} "
                   f"repair_level={result['repair_level']} problems={g['problems']}")
        u = result["usage"]
        st.caption(f"{u['llm_calls']} LLM calls, {u['tool_calls']} tools, "
                   f"{u['prompt_tokens']}+{u['completion_tokens']} tokens, {u['seconds']} s")


for turn, past in enumerate(st.session_state["history"]):
    with st.chat_message("user"):
        st.write(past["question"])
    with st.chat_message("assistant"):
        show(past, turn)

question = st.chat_input("Tanya tentang kompensasi, headcount, turnover ...") or st.session_state.pop("pending", None)
if question:
    role = st.session_state.get("role", "chro")
    with st.chat_message("user"):
        st.write(question)
    result = None
    with st.chat_message("assistant"), st.spinner("Menganalisis ..."):
        if "session_id" not in st.session_state:
            st.session_state["session_id"] = f"app-{uuid.uuid4().hex[:12]}"
        try:
            result = answer_question(question, get_client(st.session_state.get("provider")), role=role,
                                     trace={"session_id": st.session_state["session_id"], "tags": ["app"]}).to_dict()
        except Exception as e:  # noqa: BLE001 - show a readable message instead of a traceback
            st.error(f"Model tidak bisa dihubungi atau terjadi error ({type(e).__name__}). "
                     "Cek `make serve` / `vllm_server status` dan `logs/vllm.log`.")
        if result:
            show(result, len(st.session_state["history"]))
    if result:
        st.session_state["history"].append(result)
