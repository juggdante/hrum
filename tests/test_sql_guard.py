import pytest

from agent.guardrails.sql_guard import SQLRejected, check_sql

OK = "select d.department, avg(f.total_compensation) from mart.fct_compensation f " \
     "join mart.dim_department d using (department_code) where f.fiscal_year = 2026 group by 1"


def test_aggregate_query_gets_group_size_and_limit():
    sql = check_sql(OK).sql
    assert "HAVING COUNT(*) >= 5" in sql
    assert sql.endswith("LIMIT 1000")


def test_existing_having_and_small_limit_are_kept():
    sql = check_sql("select job_code, count(*) from mart.fct_compensation group by 1 having count(*) > 100 limit 3").sql
    assert "COUNT(*) > 100 AND COUNT(*) >= 5" in sql
    assert sql.endswith("LIMIT 3")


def test_cte_is_allowed():
    check_sql("with x as (select fiscal_year, total_salary from mart.fct_compensation) "
              "select fiscal_year, sum(total_salary) from x group by 1")


@pytest.mark.parametrize("sql", [
    "drop table mart.fct_compensation",
    "delete from mart.dim_job",
    "update mart.dim_job set job = 'x'",
    "create table mart.x as select 1",
    "attach 'other.db' as o",
    "copy (select 1) to 'out.csv'",
    "pragma database_list",
    "select count(*) from mart.fct_compensation; drop table mart.dim_job",
    "select count(*) from stg.stg_sf_comp__compensation",
    "select count(*) from raw.sf_employee_compensation",
    "select count(*) from fct_compensation",
    "select count(*) from read_csv('/etc/passwd')",
    "select * from mart.dim_job",
    "select total_compensation from mart.fct_compensation where job_code = '0111'",
    "select employee_key, sum(total_compensation) from mart.fct_compensation group by 1",
    "select fiscal_year, count(*) from mart.fct_compensation group by 1 union all select 1, 2 from mart.dim_job",
    "not even sql",
])
def test_rejected(sql):
    with pytest.raises(SQLRejected):
        check_sql(sql)


# ---------------------------------------------------------------- explore policy (SQL Explorer page)
COLS = {"mart.fct_compensation": {"compensation_record_id", "employee_key", "fiscal_year", "total_compensation",
                                  "department_code"},
        "mart.dim_department": {"department_code", "department"}}


def test_explore_allows_row_level_and_caps_limit():
    sql = check_sql("select fiscal_year, total_compensation from mart.fct_compensation", "explore", COLS).sql
    assert "HAVING" not in sql and sql.endswith("LIMIT 5000")


@pytest.mark.parametrize("sql,expected", [
    ("select * from mart.fct_compensation limit 5", "* EXCLUDE (compensation_record_id, employee_key)"),
    ("select f.*, d.department from mart.fct_compensation f join mart.dim_department d using (department_code)",
     "f.* EXCLUDE (compensation_record_id, employee_key)"),
    ("select * from mart.dim_department", "SELECT * FROM mart.dim_department"),
    ("select * from (select * from mart.fct_compensation) limit 1",
     "SELECT * FROM (SELECT * EXCLUDE (compensation_record_id, employee_key)"),
])
def test_explore_star_never_returns_blocked_columns(sql, expected):
    assert expected in check_sql(sql, "explore", COLS).sql


@pytest.mark.parametrize("sql", [
    "select employee_key from mart.fct_compensation",
    "select * from raw.sf_employee_compensation",
    "select count(*) from stg.stg_sf_comp__compensation",
    "select * from read_csv('/etc/passwd')",
    "drop table mart.dim_job",
    "copy (select 1) to 'out.csv'",
    "select 1 from mart.dim_job; select 2 from mart.dim_job",
    "select * from (select employee_key from mart.fct_compensation)",
])
def test_explore_still_rejects(sql):
    with pytest.raises(SQLRejected):
        check_sql(sql, "explore", COLS)


def test_explore_star_without_schema_is_rejected():
    with pytest.raises(SQLRejected):
        check_sql("select * from mart.dim_department", "explore")
