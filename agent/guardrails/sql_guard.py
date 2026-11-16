"""SQL guard for free-form SQL: the agent's `run_sql` tool and the app's SQL Explorer page.

Upgrade of the course `penjaga_sql` (P10/P11), which used regexes, to an AST check with sqlglot.

policy="agent" (run_sql, analyst role):
  1. exactly one statement, and it is a SELECT (CTEs / UNION allowed, no DDL/DML/PRAGMA/ATTACH/COPY)
  2. every table reference is in an allowed schema (`mart`), CTE names excepted; no file/system functions
  3. no blocked person-level columns (employee_key, ...) and no SELECT *
  4. the outermost SELECT must aggregate (no row-level output = no individual lookup)
  5. rewrite: HAVING count(*) >= min_group_size on the outermost SELECT, and LIMIT <= max_rows
policy="explore" (SQL Explorer page, a human analyst; decision recorded in docs/governance.md):
  rules 1-2 and the blocked columns stay; row-level output is allowed (no aggregate / group-size rule);
  SELECT * is rewritten to `* EXCLUDE (<blocked columns>)`; LIMIT <= explore_max_rows.
Execution then happens on a read_only DuckDB connection (defence in depth).
Residual risk (documented in docs/governance.md): crafted sub-queries can still narrow a group;
the CHRO role therefore only gets the semantic-layer tool, never run_sql.
"""
from __future__ import annotations

from dataclasses import dataclass

import sqlglot
from sqlglot import exp

from common.config import load_config

FORBIDDEN = (exp.Insert, exp.Update, exp.Delete, exp.Drop, exp.Create, exp.Alter, exp.Command,
             exp.Merge, exp.TruncateTable, exp.Copy, exp.Attach, exp.Detach, exp.Pragma, exp.Set,
             exp.Use, exp.Transaction, exp.Commit, exp.Rollback, exp.Grant)
FORBIDDEN_FUNCS = {"read_csv", "read_csv_auto", "read_parquet", "read_json", "read_json_auto", "read_text",
                   "read_blob", "glob", "query", "query_table", "sniff_csv", "duckdb_settings", "getenv"}


class SQLRejected(ValueError):
    """Raised when a query violates the guard. The message is shown to the LLM so it can repair."""


@dataclass
class GuardedSQL:
    sql: str          # rewritten SQL that will actually run
    tables: list[str]


def _settings(policy: str) -> tuple[set[str], set[str], int, int]:
    g = load_config("guardrails")
    max_rows = g["sql"]["explore_max_rows"] if policy == "explore" else g["sql"]["max_rows"]
    return (set(g["sql"]["allowed_schemas"]), {c.lower() for c in g["privacy"]["blocked_columns"]},
            int(g["privacy"]["min_group_size"]), int(max_rows))


def _in_star(node: exp.Expression) -> bool:
    """True for a column named inside `* EXCLUDE (...)`: it is being removed, not selected."""
    return node.find_ancestor(exp.Star) is not None


def _exclude_blocked(tree: exp.Expression, blocked: set[str],
                     table_columns: dict[str, set[str]] | None) -> None:
    """explore policy: every SELECT * / t.* gets EXCLUDE (blocked columns that its tables actually have)."""
    stars = [s for s in tree.find_all(exp.Star) if not isinstance(s.parent, exp.Count)]
    if stars and table_columns is None:
        raise SQLRejected("SELECT * needs the table schema; list the columns explicitly")
    for star in stars:
        select = star.find_ancestor(exp.Select)
        # only tables in this SELECT's own FROM/JOIN: a sub-query already handled its own star
        tables = {t.alias_or_name.lower(): f"{t.db}.{t.name}".lower()
                  for t in (select.find_all(exp.Table) if select else [])
                  if t.db and t.find_ancestor(exp.Select) is select}
        qualifier = star.parent.table.lower() if isinstance(star.parent, exp.Column) and star.parent.table else ""
        sources = [tables[qualifier]] if qualifier in tables else list(tables.values())
        present = sorted({c for src in sources for c in (table_columns or {}).get(src, set()) if c in blocked})
        if present:
            already = {c.name.lower() for c in star.args.get("except_") or []}
            star.set("except_", [*(star.args.get("except_") or []),
                                 *(exp.column(c) for c in present if c not in already)])


def _outer_select(tree: exp.Expression) -> list[exp.Select]:
    if isinstance(tree, exp.Select):
        return [tree]
    if isinstance(tree, exp.SetOperation):  # UNION / INTERSECT / EXCEPT: every branch must aggregate
        return [s for side in (tree.left, tree.right) for s in _outer_select(side)]
    if isinstance(tree, exp.Subquery):
        return _outer_select(tree.this)
    raise SQLRejected("only SELECT queries are allowed")


def check_sql(sql: str, policy: str = "agent",
              table_columns: dict[str, set[str]] | None = None) -> GuardedSQL:
    """Validate and rewrite `sql`. `table_columns` ("mart.table" -> column names) lets the explore policy
    expand SELECT * safely; the agent policy rejects SELECT * outright."""
    if policy not in ("agent", "explore"):
        raise ValueError(f"unknown SQL policy '{policy}'")
    allowed_schemas, blocked_cols, min_group, max_rows = _settings(policy)
    try:
        statements = [s for s in sqlglot.parse(sql, read="duckdb") if s is not None]
    except sqlglot.errors.ParseError as e:
        raise SQLRejected(f"SQL could not be parsed: {str(e)[:200]}") from e
    if len(statements) != 1:
        raise SQLRejected("exactly one SQL statement per call")
    tree = statements[0]
    if any(isinstance(node, FORBIDDEN) for node in tree.walk()):
        raise SQLRejected("only read-only SELECT queries are allowed")

    funcs = {(f.name or f.sql_name()).lower() for f in tree.find_all(exp.Func)}
    funcs |= {a.name.lower() for a in tree.find_all(exp.Anonymous)}
    if funcs & FORBIDDEN_FUNCS:
        raise SQLRejected(f"function(s) not allowed: {sorted(funcs & FORBIDDEN_FUNCS)}")

    cte_names = {c.alias_or_name.lower() for c in tree.find_all(exp.CTE)}
    tables = []
    for t in tree.find_all(exp.Table):
        name = t.name.lower()
        if not t.db and name in cte_names:
            continue
        if not name:
            raise SQLRejected("table functions are not allowed; query mart tables directly")
        if (t.db or "").lower() not in allowed_schemas or t.catalog:
            raise SQLRejected(f"table '{t.sql()}' is not allowed; use schema-qualified tables in {sorted(allowed_schemas)}")
        tables.append(f"{t.db}.{t.name}")

    for col in tree.find_all(exp.Column):
        if col.name.lower() in blocked_cols and not _in_star(col):
            raise SQLRejected(f"column '{col.name}' is person-level and blocked; use aggregates over dimensions")

    if policy == "explore":
        _outer_select(tree)                       # still must be a SELECT
        _exclude_blocked(tree, blocked_cols, table_columns)
    elif any(not isinstance(s.parent, exp.Count) for s in tree.find_all(exp.Star)):  # count(*) is fine
        raise SQLRejected("SELECT * is not allowed; select aggregates over named dimensions")

    for select in _outer_select(tree) if policy == "agent" else []:
        if not any(proj.find(exp.AggFunc) for proj in select.expressions):
            raise SQLRejected("the outer SELECT must aggregate (count/sum/avg/...): individual-level rows are blocked")
        size = exp.GTE(this=exp.Count(this=exp.Star()), expression=exp.Literal.number(min_group))
        having = select.args.get("having")
        select.set("having", exp.Having(this=exp.and_(having.this, size) if having else size))

    limit = tree.args.get("limit")
    current = None
    if limit is not None:
        lit = limit.expression if isinstance(limit, exp.Limit) else None
        current = int(lit.name) if isinstance(lit, exp.Literal) and lit.is_int else None
    if current is None or current > max_rows:
        tree = tree.limit(max_rows, copy=False)
    return GuardedSQL(sql=tree.sql(dialect="duckdb"), tables=sorted(set(tables)))
