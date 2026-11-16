"""Charts for agent answers and the SQL Explorer, built by code from evidence rows (never by the LLM).

Every number drawn comes from a tool result the answer cites, so a chart can never disagree with the grounded
text. Pure functions (no Streamlit) so the choice of chart is unit-tested.

Form (dataviz method): one number -> KPI tile; a year series -> line; one category -> sorted horizontal bar;
year x category -> one line per category (max 8, the palette size); two categories -> one-hue heatmap.
One y-axis per chart: metrics with different units become separate charts, never a dual axis.
Colours: the validated reference palette (categorical order fixed, light and dark steps).
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Any

import pandas as pd
import plotly.graph_objects as go

# Reference palette (dataviz skill references/palette.md): categorical slots in fixed order + blue ramp.
CATEGORICAL = {
    "light": ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"],
    "dark": ["#3987e5", "#d95926", "#199e70", "#c98500", "#d55181", "#008300", "#9085e9", "#e66767"],
}
SEQUENTIAL_BLUE = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
MAX_BARS = 15
MAX_LINES = 8          # palette size; more categories fold into "not shown" (noted in the caption)
YEAR_COLUMNS = re.compile(r"(^|_)(fiscal_)?year$")


@dataclass
class ChartSpec:
    kind: str                                   # "kpi" | "figure"
    title: str
    caption: str = ""
    figure: go.Figure | None = None
    kpis: list[tuple[str, str]] = field(default_factory=list)   # (label, formatted value)


# ------------------------------------------------------------------ formatting

def format_value(v: Any, fmt: str) -> str:
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "n/a"
    v = float(v)
    if fmt == "usd":
        for div, suffix in ((1e9, "B"), (1e6, "M"), (1e3, "K")):
            if abs(v) >= div:
                return f"${v / div:,.2f}{suffix}"
        return f"${v:,.2f}"
    if fmt == "percent":
        return f"{v * 100:.1f}%"
    if fmt == "ratio":
        return f"{v:.3f}"
    if fmt == "integer":
        return f"{v:,.0f}"
    return f"{v:,.2f}"


def _axis(values: pd.Series, fmt: str) -> tuple[float, str, str]:
    """(divisor, tickformat, unit for the axis title) for one axis. USD is scaled to K/M/B by the largest value,
    once per axis, so every series on the axis shares the same scale."""
    if fmt == "usd":
        peak = float(values.abs().max() or 0)
        for div, word in ((1e9, "billions"), (1e6, "millions"), (1e3, "thousands")):
            if peak >= div:
                return div, ",.2f", f"USD {word}"
        return 1.0, ",.0f", "USD"
    return 1.0, {"percent": ".1%", "ratio": ".3f", "integer": ",d"}.get(fmt, ",.2f"), ""


def _palette(dark: bool) -> list[str]:
    return CATEGORICAL["dark" if dark else "light"]


def _humanize(name: str) -> str:
    return name.replace("_", " ").capitalize()


def _layout(fig: go.Figure, title: str, height: int = 360) -> go.Figure:
    fig.update_layout(title={"text": title, "x": 0, "xanchor": "left", "font": {"size": 15}},
                      height=height, margin={"l": 10, "r": 10, "t": 50, "b": 10}, barcornerradius=4,
                      hovermode="closest", legend={"orientation": "h", "y": -0.18, "x": 0})
    return fig


# ------------------------------------------------------------------ building blocks

def line_chart(df: pd.DataFrame, x: str, y: str, fmt: str, title: str, color: str | None = None,
               dark: bool = False) -> go.Figure:
    fig = go.Figure()
    div, tickformat, unit = _axis(df[y], fmt)
    groups = [(None, df)] if color is None else list(df.groupby(color, sort=False))
    palette = _palette(dark)
    for i, (name, part) in enumerate(groups):
        part = part.sort_values(x)
        fig.add_trace(go.Scatter(
            x=part[x], y=part[y] / div, mode="lines+markers", name=str(name) if name is not None else _humanize(y),
            line={"width": 2, "color": palette[i % len(palette)]}, marker={"size": 8},
            customdata=[format_value(v, fmt) for v in part[y]],
            hovertemplate=f"%{{x}}<br>{_humanize(y)}: %{{customdata}}<extra>%{{fullData.name}}</extra>"))
    fig.update_yaxes(tickformat=tickformat, title=unit or None, rangemode="tozero")
    fig.update_xaxes(dtick=1 if x and YEAR_COLUMNS.search(x) else None, title=None)
    fig.update_layout(showlegend=color is not None)
    return _layout(fig, title)


def bar_chart(df: pd.DataFrame, category: str, value: str, fmt: str, title: str, dark: bool = False) -> go.Figure:
    part = df.sort_values(value, ascending=True).tail(MAX_BARS)      # largest at the top
    div, tickformat, unit = _axis(part[value], fmt)
    fig = go.Figure(go.Bar(
        x=part[value] / div, y=part[category].astype(str), orientation="h", marker={"color": _palette(dark)[0]},
        customdata=[format_value(v, fmt) for v in part[value]],
        hovertemplate="%{y}<br>%{customdata}<extra></extra>"))
    fig.update_xaxes(tickformat=tickformat, title=unit or None, rangemode="tozero")
    fig.update_yaxes(title=None, automargin=True)
    return _layout(fig, title, height=max(260, 32 * len(part) + 110))


def heatmap(df: pd.DataFrame, rows: str, cols: str, value: str, fmt: str, title: str) -> go.Figure:
    grid = df.pivot_table(index=rows, columns=cols, values=value, aggfunc="sum")
    text = grid.map(lambda v: format_value(v, fmt))
    fig = go.Figure(go.Heatmap(
        z=grid.values, x=[str(c) for c in grid.columns], y=[str(i) for i in grid.index],
        colorscale=[[i / (len(SEQUENTIAL_BLUE) - 1), c] for i, c in enumerate(SEQUENTIAL_BLUE)],
        customdata=text.values, hovertemplate="%{y} · %{x}<br>%{customdata}<extra></extra>", xgap=2, ygap=2))
    return _layout(fig, title, height=max(300, 26 * len(grid.index) + 120))


# ------------------------------------------------------------------ evidence -> charts

def _metric_meta(catalog: dict[str, Any], metric: str) -> tuple[str, bool, str]:
    """(format, synthetic, description) for a governed metric."""
    m = catalog["metrics"].get(metric, {})
    model = catalog["models"].get(m.get("model", ""), {})
    return m.get("format", "number"), bool(m.get("synthetic") or model.get("is_synthetic")), m.get("description", "")


def _title(metric: str, synthetic: bool, suffix: str = "") -> str:
    return f"{_humanize(metric)}{suffix}{' (SYNTHETIC)' if synthetic else ''}"


def charts_for_query_metric(ev: dict[str, Any], catalog: dict[str, Any], dark: bool = False) -> list[ChartSpec]:
    args, rows = ev.get("arguments") or {}, ev.get("rows") or []
    if not rows:
        return []
    df = pd.DataFrame(rows).drop(columns=["group_size"], errors="ignore")
    metrics = [m for m in args.get("metrics", []) if m in df.columns]
    dims = [d for d in (args.get("dimensions") or []) if d in df.columns]
    source = f"Source: {ev['id']} query_metric"
    if not metrics:
        return []

    if not dims or len(df) == 1:   # one number per metric: stat tiles
        label_prefix = " · ".join(str(df.iloc[0][d]) for d in dims)
        kpis = []
        for m in metrics:
            fmt, synthetic, _ = _metric_meta(catalog, m)
            label = f"{_humanize(m)}{' (SYNTHETIC)' if synthetic else ''}"
            kpis.append((f"{label_prefix} · {label}" if label_prefix else label, format_value(df.iloc[0][m], fmt)))
        return [ChartSpec("kpi", "", source, kpis=kpis)]

    specs = []
    year = next((d for d in dims if YEAR_COLUMNS.search(d)), None)
    cats = [d for d in dims if d != year]
    if year and not cats:                                   # time series: one chart per unit
        by_fmt: dict[str, list[str]] = {}
        for m in metrics:
            by_fmt.setdefault(_metric_meta(catalog, m)[0], []).append(m)
        for fmt, group in by_fmt.items():
            long = df.melt(id_vars=[year], value_vars=group, var_name="metric", value_name="value") \
                if len(group) > 1 else None
            synthetic = any(_metric_meta(catalog, m)[1] for m in group)
            title = _title(group[0] if len(group) == 1 else " vs ".join(group), synthetic, " by fiscal year")
            fig = line_chart(long if long is not None else df, year, "value" if long is not None else group[0],
                             fmt, title, color="metric" if long is not None else None, dark=dark)
            specs.append(ChartSpec("figure", title, f"{_metric_meta(catalog, group[0])[2]} {source}", fig))
        return specs

    m = metrics[0]                                          # categorical forms chart the first metric
    fmt, synthetic, desc = _metric_meta(catalog, m)
    note = ""
    if year and len(cats) == 1:
        totals = df.groupby(cats[0])[m].sum().sort_values(ascending=False)
        keep = totals.index[:MAX_LINES]
        if len(totals) > MAX_LINES:
            note = f" Showing the top {MAX_LINES} of {len(totals)} {cats[0]} values."
        title = _title(m, synthetic, f" by fiscal year and {cats[0]}")
        fig = line_chart(df[df[cats[0]].isin(keep)], year, m, fmt, title, color=cats[0], dark=dark)
    elif len(cats) == 1:
        if len(df) > MAX_BARS:
            note = f" Showing the top {MAX_BARS} of {len(df)}."
        title = _title(m, synthetic, f" by {cats[0]}")
        fig = bar_chart(df, cats[0], m, fmt, title, dark=dark)
    elif len(cats) == 2 and not year:
        title = _title(m, synthetic, f" by {cats[0]} and {cats[1]}")
        fig = heatmap(df, cats[0], cats[1], m, fmt, title)
    else:
        return []
    return [ChartSpec("figure", title, f"{desc}{note} {source}".strip(), fig)]


def charts_for_attrition(ev: dict[str, Any], dark: bool = False) -> list[ChartSpec]:
    rows = ev.get("rows") or []
    if not rows or "avg_predicted_risk" not in rows[0]:
        return []
    df = pd.DataFrame(rows)
    source = f"Source: {ev['id']} attrition_insight"
    if len(df) == 1:   # one segment: a single bar says nothing, show tiles instead
        r = df.iloc[0]
        kpis = [(f"{r['segment']} · predicted risk (SYNTHETIC)", format_value(r["avg_predicted_risk"], "percent"))]
        for col, label in (("high_risk_count", "high-risk employees"), ("expected_leavers", "expected leavers")):
            if col in df:
                kpis.append((f"{r['segment']} · {label}", format_value(r[col], "integer")))
        return [ChartSpec("kpi", "", source, kpis=kpis)]
    mixed = df["segment_type"].nunique() > 1 if "segment_type" in df else False
    df["label"] = df["segment"].astype(str) + (" (" + df["segment_type"].str.replace("_", " ") + ")" if mixed else "")
    title = "Predicted attrition risk by segment (SYNTHETIC)"
    fig = bar_chart(df, "label", "avg_predicted_risk", "percent", title, dark=dark)
    return [ChartSpec("figure", title, f"Mean predicted probability of leaving in FY2026. {source}", fig)]


def chart_for_table(df: pd.DataFrame, title: str = "", source: str = "", dark: bool = False) -> ChartSpec | None:
    """Best-effort chart for arbitrary rows (run_sql evidence, SQL Explorer). None when no form fits."""
    if df.empty or len(df.columns) < 1:
        return None
    numeric = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c]) and not pd.api.types.is_bool_dtype(df[c])]
    year = next((c for c in df.columns if YEAR_COLUMNS.search(str(c).lower()) and c in numeric), None)
    values = [c for c in numeric if c != year]
    # pandas >= 3 gives text columns the "str" dtype, older versions "object": accept both
    text = [c for c in df.columns if c not in numeric
            and (pd.api.types.is_string_dtype(df[c]) or df[c].dtype == object)]
    if not values:
        return None

    def fmt(col: str) -> str:
        return "integer" if pd.api.types.is_integer_dtype(df[col]) else "number"

    if len(df) == 1 and len(values) <= 4:
        return ChartSpec("kpi", title, source, kpis=[(_humanize(str(c)), format_value(df.iloc[0][c], fmt(c)))
                                                     for c in values])
    v = values[0]
    if year and not text and df[year].nunique() == len(df):
        name = title or f"{_humanize(v)} by {year}"
        return ChartSpec("figure", name, source, line_chart(df, year, v, fmt(v), name, dark=dark))
    if year and len(text) == 1:
        name = title or f"{_humanize(v)} by {year} and {text[0]}"
        top = df.groupby(text[0])[v].sum().sort_values(ascending=False).index[:MAX_LINES]
        return ChartSpec("figure", name, source, line_chart(df[df[text[0]].isin(top)], year, v, fmt(v),
                                                            name, color=text[0], dark=dark))
    if len(text) == 1 and df[text[0]].nunique() == len(df) and len(df) <= 200:
        name = title or f"{_humanize(v)} by {text[0]}"
        return ChartSpec("figure", name, source, bar_chart(df, text[0], v, fmt(v), name, dark=dark))
    return None


def charts_for_evidence(evidence: list[dict[str, Any]], catalog: dict[str, Any], dark: bool = False,
                        cited_only: bool = True) -> list[ChartSpec]:
    specs: list[ChartSpec] = []
    for ev in evidence:
        if cited_only and not ev.get("cited"):
            continue
        if ev["tool"] == "query_metric":
            specs += charts_for_query_metric(ev, catalog, dark)
        elif ev["tool"] == "attrition_insight":
            specs += charts_for_attrition(ev, dark)
        elif ev["tool"] == "run_sql" and ev.get("rows"):
            spec = chart_for_table(pd.DataFrame(ev["rows"]), source=f"Source: {ev['id']} run_sql", dark=dark)
            specs += [spec] if spec else []
    return specs
