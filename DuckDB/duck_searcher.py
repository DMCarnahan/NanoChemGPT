from __future__ import annotations

import os
from typing import List, Optional
import duckdb
import pandas as pd

parq = os.getenv("LOOKUP_PARQUET_GLOB")         # e.g. /mnt/data/datasets/**/*.parquet
dbp  = os.getenv("LOOKUP_DUCKDB_PATH")          # e.g. /mnt/data/datasets.duckdb
tbl  = os.getenv("LOOKUP_DUCKDB_TABLE") or "reactions"

if dbp and parq and not os.path.exists(dbp):
    con = duckdb.connect(dbp)
    con.execute(f"CREATE TABLE {tbl} AS SELECT * FROM read_parquet('{parq}', hive_partitioning=1)")
    con.execute("CHECKPOINT")
    con.close()

DEFAULT_SELECT = ["title", "year", "url", "doi", "procedure", "solvent", "notes"]

class DuckSearcher:
    """
    Query synthesis/reaction rows using DuckDB, either over:
      - a VIEW that scans Parquet/CSV via read_parquet/read_csv_auto
      - a physical table inside a .duckdb database
    """
    def __init__(
        self,
        con: duckdb.DuckDBPyConnection,
        source_sql: str,
        text_cols: List[str],
        select_cols: Optional[List[str]] = None,
        memory_limit: Optional[str] = "1GB",
        view_name: str = "reactions_view",
    ) -> None:
        self.con = con
        self.view = view_name
        self.text_cols = [c for c in (text_cols or []) if c]
        self.select_cols = list(select_cols or DEFAULT_SELECT)
        if memory_limit:
            try:
                self.con.execute(f"SET memory_limit='{memory_limit}'")
            except Exception:
                pass 
        # Create/Replace the view so subsequent queries are stable
        self.con.execute(f"CREATE OR REPLACE VIEW {self.view} AS {source_sql}")

    def query(self, pattern: str, topk: int = 8, **_) -> pd.DataFrame:
        """
        Case-insensitive substring search across configured text columns.
        Returns a Pandas DataFrame with selected columns.
        """
        if not (pattern or "").strip():
            return pd.DataFrame(columns=self.select_cols)

        like = "%" + pattern.replace("%", " ").replace("_", " ") + "%"
        where = " OR ".join([f"coalesce({c}, '') ILIKE ?" for c in self.text_cols]) or "TRUE"
        sql = f"SELECT {', '.join(self.select_cols)} FROM {self.view} WHERE {where} LIMIT ?"
        bind = [like] * (where.count("ILIKE")) + [int(topk)]
        return self.con.execute(sql, bind).df()


def get_duck_searcher():
    """
    Env-driven factory. Supports three modes (checked in this order):
      1) LOOKUP_PARQUET_GLOB → scan parquet files
      2) LOOKUP_CSV_GLOB     → scan csv files
      3) LOOKUP_DUCKDB_PATH + LOOKUP_DUCKDB_TABLE → open DB/table

    Other envs:
      - LOOKUP_TEXT_COLS      (comma list; default: procedure,solvent,notes,title)
      - LOOKUP_SELECT_COLS    (comma list; default: title,year,url,doi,procedure,solvent,notes)
      - LOOKUP_MEMORY_LIMIT   (e.g., 1GB)
    """
    parq_glob = os.getenv("LOOKUP_PARQUET_GLOB")
    csv_glob  = os.getenv("LOOKUP_CSV_GLOB")
    db_path   = os.getenv("LOOKUP_DUCKDB_PATH") or ":memory:"
    db_table  = os.getenv("LOOKUP_DUCKDB_TABLE")

    text_cols = [c.strip() for c in (os.getenv("LOOKUP_TEXT_COLS") or "procedure,solvent,notes,title").split(",") if c.strip()]
    select_cols = [c.strip() for c in (os.getenv("LOOKUP_SELECT_COLS") or ",".join(DEFAULT_SELECT)).split(",") if c.strip()]
    mem = os.getenv("LOOKUP_MEMORY_LIMIT", "1GB")

    con = duckdb.connect(db_path)

    if parq_glob:
        src = f"SELECT * FROM read_parquet('{parq_glob}', hive_partitioning=1)"
        return DuckSearcher(con, src, text_cols, select_cols, memory_limit=mem)

    if csv_glob:
        src = f"SELECT * FROM read_csv_auto('{csv_glob}', header=1, union_by_name=true)"
        return DuckSearcher(con, src, text_cols, select_cols, memory_limit=mem)

    if db_table:
        src = f"SELECT * FROM {db_table}"
        return DuckSearcher(con, src, text_cols, select_cols, memory_limit=mem)

    return None
