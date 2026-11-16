# Architecture
SF Comp (real) + OEWS/H-1B (benchmark, later) + synthetic HRIS
  -> ingestion -> BigQuery raw -> dbt (staging -> marts + semantic layer)
  -> dashboard (Looker Studio/Streamlit) | ML (attrition, pay equity, forecast) | Agent (text-to-metric/SQL)
TODO: replace with a diagram.
