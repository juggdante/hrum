"""Download SF Employee Compensation and load it AS-IS into DuckDB raw.sf_employee_compensation.

Raw layer rules (CLAUDE.md): every column VARCHAR, no cleaning, plus `_loaded_at` and `_source_file`.
The load is idempotent: the table is replaced on every run.
"""
from __future__ import annotations

import logging
import os
import shutil
from datetime import date
from pathlib import Path

import duckdb
import requests

from common.config import ROOT, db_path, load_config

log = logging.getLogger(__name__)
RAW_DIR = ROOT / "data" / "raw"
EXPECTED_COLUMNS = 26


def fetch_csv(force: bool = False) -> Path:
    """Return a local CSV path: $SF_COMP_CSV if set, else a dated download into data/raw/."""
    local = os.environ.get("SF_COMP_CSV")
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    if local:
        src = Path(local).expanduser()
        dst = RAW_DIR / src.name
        if not dst.exists() or force:
            shutil.copy2(src, dst)
        return dst
    dst = RAW_DIR / f"Employee_Compensation_{date.today():%Y%m%d}.csv"
    if dst.exists() and not force:
        return dst
    url = load_config("platform")["sources"]["sf_comp"]["csv_url"]
    log.info("Downloading %s", url)
    with requests.get(url, stream=True, timeout=120) as r:
        r.raise_for_status()
        tmp = dst.with_suffix(".part")
        with open(tmp, "wb") as f:
            f.writelines(r.iter_content(chunk_size=1 << 20))
        tmp.rename(dst)
    return dst


def load_raw(csv_path: Path, con: duckdb.DuckDBPyConnection) -> int:
    """Load the CSV into raw.<table> as VARCHAR columns. Returns the row count."""
    cfg = load_config("platform")
    schema = cfg["schemas"]["raw"]
    table = f"{schema}.{cfg['sources']['sf_comp']['raw_table']}"
    con.execute(f"CREATE SCHEMA IF NOT EXISTS {schema}")
    con.execute(
        f"""
        CREATE OR REPLACE TABLE {table} AS
        SELECT *, current_timestamp AS _loaded_at, $name AS _source_file
        FROM read_csv($path, header = true, all_varchar = true)
        """,
        {"name": csv_path.name, "path": str(csv_path)},
    )
    n_cols = len(con.execute(f"DESCRIBE {table}").fetchall()) - 2
    if n_cols != EXPECTED_COLUMNS:
        raise ValueError(f"Expected {EXPECTED_COLUMNS} source columns, got {n_cols}")
    return con.execute(f"SELECT count(*) FROM {table}").fetchone()[0]


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    csv_path = fetch_csv()
    path = db_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    with duckdb.connect(str(path)) as con:
        n = load_raw(csv_path, con)
    log.info("Loaded %s rows from %s into %s", f"{n:,}", csv_path.name, path)


if __name__ == "__main__":
    main()
