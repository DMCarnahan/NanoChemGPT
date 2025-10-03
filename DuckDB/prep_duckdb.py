"""
Prepare Parquet files and/or a DuckDB database from raw datasets.
Usage examples:

# Convert JSONL/JSON → a single Parquet:
python prep_duckdb.py --json-glob "/mnt/data/raw/**/*.jsonl" --to-parquet "/mnt/data/reactions.parquet"

# Convert CSVs → a single Parquet:
python prep_duckdb.py --csv-glob "/mnt/data/raw/**/*.csv"   --to-parquet "/mnt/data/reactions.parquet"

# Build a .duckdb database with a physical table from Parquet:
python prep_duckdb.py --from-parquet "/mnt/data/reactions.parquet" \
    --duckdb "/mnt/data/nanochem.duckdb" --table reactions

Requires: duckdb>=0.9, pandas
"""

from __future__ import annotations

import argparse

import duckdb


def build_parquet_from_json(json_glob: str, out_parquet: str) -> None:
    con = duckdb.connect()
    con.execute("INSTALL json; LOAD json;")
    sql = f"COPY (SELECT * FROM read_json_auto('{json_glob}')) TO '{out_parquet}' (FORMAT PARQUET);"
    con.execute(sql)


def build_parquet_from_csv(csv_glob: str, out_parquet: str) -> None:
    con = duckdb.connect()
    sql = f"COPY (SELECT * FROM read_csv_auto('{csv_glob}', header=1, union_by_name=true)) TO '{out_parquet}' (FORMAT PARQUET);"
    con.execute(sql)


from pathlib import Path


def make_duckdb_from_parquet(parquet_path: str, duckdb_path: str, table: str) -> None:
    p_parq = Path(parquet_path).resolve()
    p_db = Path(duckdb_path).resolve()
    p_db.parent.mkdir(parents=True, exist_ok=True)  # ensure folder exists
    print(f"[prep] ABS paths: parquet={p_parq} db={p_db}")

    con = duckdb.connect(str(p_db))
    # Use forward slashes inside SQL for portability
    parq_sql = str(p_parq).replace("\\", "/")
    con.execute(
        f"CREATE OR REPLACE TABLE {table} AS SELECT * FROM read_parquet('{parq_sql}')"
    )
    con.execute("CHECKPOINT")  # flush to disk
    con.close()
    print(f"[prep] wrote DB? {p_db.exists()} → {p_db}")


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="Prepare Parquet and/or DuckDB from raw datasets."
    )
    ap.add_argument(
        "--json-glob", default=None, help="Glob of JSON/JSONL files to convert"
    )
    ap.add_argument("--csv-glob", default=None, help="Glob of CSV files to convert")
    ap.add_argument("--to-parquet", default=None, help="Output Parquet file")
    ap.add_argument(
        "--from-parquet", default=None, help="Existing Parquet to import into DuckDB"
    )
    ap.add_argument(
        "--duckdb", default=None, help="Output DuckDB path to create/update"
    )
    ap.add_argument(
        "--table", default="reactions", help="DuckDB table name (default reactions)"
    )
    args = ap.parse_args(argv)

    if args.json_glob and args.to_parquet:
        print(f"[prep] JSON → Parquet: {args.json_glob} -> {args.to_parquet}")
        build_parquet_from_json(args.json_glob, args.to_parquet)

    if args.csv_glob and args.to_parquet:
        print(f"[prep] CSV  → Parquet: {args.csv_glob} -> {args.to_parquet}")
        build_parquet_from_csv(args.csv_glob, args.to_parquet)

    parquet_src = args.from_parquet or args.to_parquet
    if parquet_src and args.duckdb:
        print(
            f"[prep] Parquet → DuckDB table '{args.table}': {parquet_src} -> {args.duckdb}"
        )
        make_duckdb_from_parquet(parquet_src, args.duckdb, args.table)

    if not (args.json_glob or args.csv_glob or (args.from_parquet and args.duckdb)):
        ap.print_help()
        return 2

    print("[prep] done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
