"""Question bank: the verified golden questions (evals/golden_questions.yaml), grouped, one click to ask."""
from __future__ import annotations

import streamlit as st

from agent.app.data import catalog, eval_results, golden_questions

SECTIONS = [  # category -> (heading, explanation)
    ("cost", "Biaya kompensasi (data REAL)", "Total, rata-rata, median, gaji vs benefit, overtime, tren antar tahun."),
    ("fairness", "Keadilan gaji (compa-ratio)", "Gaji dibanding median jabatan yang sama (benchmark internal)."),
    ("risk", "Risiko turnover (data SINTETIS + model ML)",
     "Turnover, biaya turnover, masa kerja, prediksi risiko attrition dan pendorongnya."),
    ("docs", "Definisi & cara membaca data", "Dijawab dari dokumentasi HRUM (FAQ, definisi metric)."),
    ("analyst", "Analyst: SQL langsung", "Untuk role analyst: agent menulis SQL agregat ke tabel mart."),
    ("abstain", "Di luar cakupan data", "HRUM akan bilang tidak bisa menjawab, bukan menebak."),
    ("attack", "Yang akan ditolak HRUM", ("Privasi dan keamanan: data individu, nama, perubahan data, "
                                          "instruksi untuk melanggar aturan.")),
]

TEMPLATES = [
    "Berapa <metric> di FY2026?",
    "Bagaimana tren <metric> dari FY2020 sampai FY2026?",
    "<Department / organization group / job family> mana dengan <metric> tertinggi di FY2026?",
    "Bandingkan <metric> antara <department A> dan <department B> di FY2026.",
    "Berapa persen kenaikan <metric> dari FY2025 ke FY2026?",
    "Berapa <metric> per <level / age band / gender> (sintetis)?",
    "Apa arti <metric> dan bagaimana menghitungnya?",
]

st.title("Daftar pertanyaan")
results = eval_results()
passed = results.get("per_question", {})
if results:
    acc = results["accuracy"]
    st.caption(f"Semua pertanyaan di bawah adalah golden questions yang diuji otomatis. Hasil terakhir "
               f"({results['served']['model']}, prompt {results.get('prompt_version', '?')}): "
               f"{acc['k']}/{acc['n']} lulus. Klik **Tanya** untuk mengirimnya ke chat.")

by_cat: dict[str, list[dict]] = {}
for q in golden_questions():
    by_cat.setdefault(q["category"], []).append(q)

for cat, heading, explanation in SECTIONS:
    items = by_cat.get(cat, [])
    if not items:
        continue
    st.subheader(heading)
    st.caption(explanation)
    for q in items:
        left, right = st.columns([6, 1], vertical_alignment="center")
        role_note = " · role analyst" if q.get("role") == "analyst" else ""
        mark = " · :green[lulus eval]" if passed.get(q["id"]) else ""
        left.markdown(f"{q['question']}  \n:gray[{q['id']}{role_note}]{mark}")
        if right.button("Tanya", key=f"ask-{q['id']}", width="stretch"):
            st.session_state["pending"] = q["question"]
            st.session_state["role_request"] = q.get("role", "chro")
            st.switch_page("views/chat.py")

st.subheader("Template pertanyaan")
st.caption("Ganti bagian <...> dengan metric dan dimensi dari daftar di bawah atau dari Panduan dataset.")
st.markdown("\n".join(f"- {t}" for t in TEMPLATES))
cat = catalog()
with st.expander("Metric yang bisa ditanyakan"):
    st.markdown("\n".join(
        f"- `{name}`{' (sintetis)' if m.get('synthetic') or cat['models'][m['model']]['is_synthetic'] else ''}: "
        f"{m['description']}" for name, m in cat["metrics"].items()))
