"""
Ingestion scaffolding for mechanistic knowledge entries.
- Reads JSONL/JSON of extracted facts
- Validates against mechanistic.schema.json
- Writes normalized JSONL and a lightweight SQLite for metadata
"""
from __future__ import annotations
import json, pathlib, sqlite3, time, uuid
from typing import Dict, Any, Iterable
import jsonschema

ROOT = pathlib.Path(__file__).resolve().parents[1]
SCHEMA_PATH = ROOT / "schemas" / "mechanistic.schema.json"
DB_PATH = ROOT / "mechanistic_kb"  / "mechanistic_meta.sqlite"
JSONL_OUT = ROOT / "mechanistic_kb" / "mechanistic.jsonl"

def _ensure_dirs():
    (ROOT / "mechanistic_kb").mkdir(parents=True, exist_ok=True)

def validate_entry(entry: Dict[str, Any]) -> Dict[str, Any]:
    schema = json.loads(SCHEMA_PATH.read_text())
    jsonschema.validate(entry, schema)
    return entry

def normalize_entry(entry: Dict[str, Any]) -> Dict[str, Any]:
    now = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    entry.setdefault("id", str(uuid.uuid4()))
    entry.setdefault("created_at", now)
    entry["updated_at"] = now
    return entry

def write_sqlite_meta(entries: Iterable[Dict[str, Any]]):
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS entries (
          id TEXT PRIMARY KEY,
          system TEXT,
          method TEXT,
          influential_param TEXT,
          scope TEXT,
          citation TEXT
        )
    """)
    for e in entries:
        citation = e.get("evidence", [{}])[0].get("citation", "") if e.get("evidence") else ""
        cur.execute("REPLACE INTO entries VALUES (?,?,?,?,?,?)",
                    (e["id"], e.get("system",""), e.get("synthesis_method",""),
                     (e.get("most_influential_parameter") or {}).get("name",""),
                     (e.get("most_influential_parameter") or {}).get("scope",""),
                     citation))
    conn.commit(); conn.close()

def append_jsonl(entries: Iterable[Dict[str, Any]]):
    with JSONL_OUT.open("a", encoding="utf-8") as f:
        for e in entries:
            f.write(json.dumps(e, ensure_ascii=False) + "\n")

def ingest(entries: Iterable[Dict[str, Any]]):
    _ensure_dirs()
    validated = [validate_entry(normalize_entry(e)) for e in entries]
    write_sqlite_meta(validated)
    append_jsonl(validated)
    return [e["id"] for e in validated]

if __name__ == "__main__":
    # Demo run with the sample entry
    sample_path = ROOT / "mechanistic_kb" / "sample_entries" / "lab6_example.json"
    sample = json.loads(sample_path.read_text())
    ids = ingest([sample])
    print("Ingested:", ids)
