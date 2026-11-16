"""SQL Explorer (analyst role): read-only SQL on the mart tables, row-level allowed, person keys blocked.

Guard: agent/guardrails/sql_guard.py policy="explore" (decision recorded in docs/governance.md).
"""
from __future__ import annotations

import duckdb
import pandas as pd
import streamlit as st

from agent.app.charts import chart_for_table
from agent.app.data import cursor, is_dark, mart_columns, mart_schema
from agent.guardrails.sql_guard import SQLRejected, check_sql
from agent.tools.sql import execute_with_timeout
from common.config import load_config

EXAMPLES = {
    "Total kompensasi per tahun": "select fiscal_year, sum(total_compensation) as total_compensation\n"
                                  "from mart.fct_compensation where is_paid\ngroup by 1 order by 1",
    "10 job dengan rata-rata kompensasi tertinggi (FY2026)":
        "select j.job, count(*) as records, avg(f.total_compensation) as avg_total_compensation\n"
        "from mart.fct_compensation f join mart.dim_job j using (job_code)\n"
        "where f.fiscal_year = 2026 and f.is_paid\ngroup by 1 having count(*) >= 5\norder by 3 desc limit 10",
    "Contoh baris pay record": "select * from mart.fct_compensation where fiscal_year = 2026 limit 20",
    "Turnover per organization group (sintetis)":
        "select d.organization_group, avg(case when s.is_terminated then 1.0 else 0.0 end) as turnover_rate\n"
        "from mart.fct_employee_snapshot s join mart.dim_department d using (department_code)\n"
        "group by 1 order by 2 desc",
}

if st.session_state.get("role") != "analyst":
    st.warning("SQL Explorer hanya untuk role analyst (pilih di sidebar).")
    st.stop()

st.title("SQL Explorer")
max_rows = load_config("guardrails")["sql"]["explore_max_rows"]
st.info(f"Read-only, hanya schema `mart`. Baris per pay record boleh (data publik), maksimal {max_rows:,} baris; "
        "kolom kunci per orang (`employee_key`, `compensation_record_id`) selalu disembunyikan. "
        "Jawaban agent tetap agregat (grup minimal 5).")

left, right = st.columns([1, 2])
with left:
    st.subheader("Tabel")
    schema = mart_schema()
    for table in schema["table"].unique():
        cols = schema[schema["table"] == table]
        with st.expander(f"mart.{table}"):
            st.dataframe(cols.assign(column=cols["column"] + cols["blocked"].map({True: " (blocked)", False: ""}))
                         [["column", "type"]], hide_index=True)

with right:
    example = st.selectbox("Contoh query", ["(pilih)", *EXAMPLES])
    if example != "(pilih)" and st.session_state.get("_last_example") != example:
        st.session_state["sql_text"] = EXAMPLES[example]
        st.session_state["_last_example"] = example
    sql = st.text_area("SQL", key="sql_text", height=180, placeholder="select ... from mart.fct_compensation ...")
    # never disabled: a text area commits on blur, so a disabled button would swallow the first click after typing
    run = st.button("Jalankan", type="primary")
    if run and not (sql or "").strip():
        st.warning("Tulis query SQL dulu.")
    elif run:
        try:
            guarded = check_sql(sql, policy="explore", table_columns=mart_columns())
            cols, rows = execute_with_timeout(cursor(), guarded.sql)
            st.session_state["sql_result"] = {"sql": guarded.sql, "df": pd.DataFrame(rows, columns=cols)}
        except SQLRejected as e:
            st.session_state.pop("sql_result", None)
            st.error(f"Query ditolak: {e}")
        except duckdb.InterruptException:
            st.session_state.pop("sql_result", None)
            st.error(f"Query dihentikan: lebih dari {load_config('guardrails')['sql']['timeout_seconds']} detik.")
        except duckdb.Error as e:
            st.session_state.pop("sql_result", None)
            st.error(f"Error DuckDB: {str(e).splitlines()[0]}")

    res = st.session_state.get("sql_result")
    if res:
        df = res["df"]
        st.caption("SQL yang dijalankan (setelah guard):")
        st.code(res["sql"], language="sql")
        note = f" (dipotong di {max_rows:,})" if len(df) >= max_rows else ""
        st.markdown(f"**{len(df):,} baris**{note}")
        spec = chart_for_table(df, dark=is_dark())
        if spec and spec.kind == "figure":
            st.plotly_chart(spec.figure, theme="streamlit", key="sql-chart")
        elif spec and spec.kind == "kpi":
            for label, value in spec.kpis:
                st.metric(label, value)
        st.dataframe(df, hide_index=True)
        st.download_button("Unduh CSV", df.to_csv(index=False).encode(), "hrum_query.csv", "text/csv")
