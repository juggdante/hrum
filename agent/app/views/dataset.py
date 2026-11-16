"""Dataset guide, generated from the sources of truth: live DuckDB stats, dbt model docs, metrics.yaml, FAQ."""
from __future__ import annotations

import pandas as pd
import streamlit as st

from agent.app.charts import chart_for_table, format_value
from agent.app.data import catalog, faq_sections, is_dark, mart_schema, query_df, table_descriptions

st.title("Panduan dataset")
st.markdown(
    "HRUM menggabungkan **data kompensasi REAL** pegawai City & County of San Francisco "
    "(SF Employee Compensation, data.sfgov.org, lisensi PDDL 1.0) dengan **lapisan HRIS SINTETIS** "
    "(tanggal masuk/keluar, level, kinerja, gender, kelompok umur) untuk mendemokan analisis turnover. "
    "Hasil apa pun dari data sintetis menunjukkan kemampuan platform, bukan temuan tentang SF.")

# ------------------------------------------------------------------ coverage (live)
st.header("Cakupan data")
stats = query_df("""
    select min(fiscal_year) as first_fy, max(fiscal_year) as last_fy,
           count(*) filter (where is_paid) as paid_records,
           count(distinct department_code) as departments, count(distinct job_code) as jobs
    from mart.fct_compensation""").iloc[0]
groups = query_df("select count(distinct organization_group) as n from mart.dim_department").iloc[0]["n"]
families = query_df("select count(distinct job_family) as n from mart.dim_job").iloc[0]["n"]
snap = query_df("""select count(*) as employees, avg(case when is_terminated then 1.0 else 0.0 end) as turnover
                   from mart.fct_employee_snapshot""").iloc[0]

c = st.columns(4)
c[0].metric("Tahun fiskal", f"FY{int(stats['first_fy'])}-FY{int(stats['last_fy'])}")
c[1].metric("Pay record (dibayar)", format_value(stats["paid_records"], "integer"))
c[2].metric("Department / org group", f"{int(stats['departments'])} / {int(groups)}")
c[3].metric("Job / job family", f"{int(stats['jobs']):,} / {int(families)}")
c = st.columns(4)
c[0].metric("Karyawan di snapshot FY2026 (sintetis)", format_value(snap["employees"], "integer"))
c[1].metric("Turnover FY2026 (sintetis)", format_value(snap["turnover"], "percent"))

per_year = query_df("select fiscal_year, count(*) filter (where is_paid) as paid_records "
                    "from mart.fct_compensation group by 1 order by 1")
spec = chart_for_table(per_year, title="Pay record yang dibayar per tahun fiskal", dark=is_dark())
if spec and spec.figure is not None:
    st.plotly_chart(spec.figure, theme="streamlit", key="paid-per-year")

# ------------------------------------------------------------------ real vs synthetic + FAQ
st.header("Cara membaca data")
for heading, body in faq_sections():
    with st.expander(heading, expanded=heading.startswith("Which data is real")):
        st.markdown(body)

# ------------------------------------------------------------------ metrics
st.header("Metric yang tersedia")
st.caption("Satu definisi per metric (`warehouse/semantic/metrics.yaml`); agent, eval dan dashboard memakai "
           "definisi yang sama.")
cat = catalog()
st.dataframe(pd.DataFrame([{
    "metric": name,
    "definition": m["description"],
    "unit": m.get("format", ""),
    "data": "SYNTHETIC" if m.get("synthetic") or cat["models"][m["model"]]["is_synthetic"] else "REAL",
    "table": cat["models"][m["model"]]["table"],
    "group / filter by": ", ".join(cat["models"][m["model"]]["dimensions"]),
} for name, m in cat["metrics"].items()]), hide_index=True)

# ------------------------------------------------------------------ tables
st.header("Tabel (schema `mart`)")
st.caption("Kolom yang ditandai *blocked* adalah kunci per orang; tidak bisa dipilih di agent maupun SQL Explorer.")
schema = mart_schema()
for table, desc in table_descriptions().items():
    cols = schema[schema["table"] == table]
    if cols.empty:
        continue
    with st.expander(f"mart.{table}  ({len(cols)} kolom)"):
        st.markdown(desc)
        st.dataframe(cols[["column", "type", "description", "blocked"]], hide_index=True)

# ------------------------------------------------------------------ dimension values
st.header("Nilai dimensi")
c1, c2 = st.columns(2)
with c1:
    st.subheader("Organization group")
    st.dataframe(query_df("select organization_group, count(*) as departments from mart.dim_department "
                          "group by 1 order by 1"), hide_index=True)
with c2:
    st.subheader("Job family (top 20 menurut jumlah job)")
    st.dataframe(query_df("select job_family, count(*) as jobs from mart.dim_job group by 1 "
                          "order by 2 desc limit 20"), hide_index=True)
with st.expander("Semua department"):
    st.dataframe(query_df("select department, organization_group from mart.dim_department order by 1"),
                 hide_index=True)
