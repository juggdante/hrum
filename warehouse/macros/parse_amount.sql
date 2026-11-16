{# Source amounts are strings with thousands separators, e.g. "79,185.04". #}
{% macro parse_amount(col) -%}
  try_cast(replace({{ adapter.quote(col) }}, ',', '') as double)
{%- endmacro %}
