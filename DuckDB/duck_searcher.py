import os
from typing import Optional, Sequence

import pandas as pd

import duckdb


class DuckSearcher:
    def __init__(self, db_path: str, table: str, text_cols: Sequence[str]):
        self.db_path = db_path
        self.table = table
        self.text_cols = list(text_cols)
        self._validate_table()

    def _validate_table(self):
        with duckdb.connect(self.db_path, read_only=True) as con:
            cols = [
                c[0]
                for c in con.execute(f"PRAGMA table_info('{self.table}')").fetchall()
            ]
            self.text_cols = [c for c in self.text_cols if c in cols]

    def query(self, pattern: str, topk: int = 8, **kwargs) -> pd.DataFrame:
        if not pattern.strip() or not self.text_cols:
            return pd.DataFrame()

        like = "%" + pattern.replace("%", " ").replace("_", " ") + "%"
        where = " OR ".join([f"{col} ILIKE ?" for col in self.text_cols])
        sql = f"SELECT * FROM {self.table} WHERE {where} LIMIT {int(topk)}"

        with duckdb.connect(self.db_path, read_only=True) as con:
            rows = con.execute(sql, [like] * len(self.text_cols)).fetchdf()
        return rows


def get_duck_searcher() -> Optional[DuckSearcher]:
    db_path = os.getenv("LOOKUP_DUCKDB_PATH")
    table = os.getenv("LOOKUP_DUCKDB_TABLE", "reactions")
    text_cols = [
        s.strip() for s in os.getenv("LOOKUP_TEXT_COLS", "").split(",") if s.strip()
    ]
    if db_path and os.path.exists(db_path) and text_cols:
        try:
            return DuckSearcher(db_path, table, text_cols)
        except Exception as e:
            print("[duck_searcher] failed to initialize:", e)
            return None
    return None
