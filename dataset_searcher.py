from __future__ import annotations
import json, gzip, lzma, re
from pathlib import Path
import pandas as pd

# ----------  I/O helpers  ----------------------------------------------------
def _read_json_array(fp, comp=None):
    opener = _opener(fp, comp)
    with opener(fp, "rt") as f:
        data = json.load(f)
    return pd.json_normalize(data if isinstance(data, list) else data.get("records", data))

def _read_jsonl(fp, comp=None):
    opener = _opener(fp, comp)
    with opener(fp, "rt") as f:
        rows = [json.loads(l) for l in f if l.strip()]
    return pd.json_normalize(rows)

def _opener(fp: Path, comp):
    return gzip.open if comp == "gzip" else lzma.open if comp == "xz" else open

def load_table(path: str | Path) -> pd.DataFrame:
    p = Path(path)
    suf = p.suffix.lower()

    match suf:
        case ".csv" | ".csv.gz":      return pd.read_csv(p)
        case ".xlsx":                 return pd.read_excel(p)
        case ".json":                 return _read_json_array(p)
        case ".jsonl":                return _read_jsonl(p)
        case ".json.gz":              return _read_json_array(p, "gzip")
        case ".jsonl.gz":             return _read_jsonl(p, "gzip")
        case ".json.xz":              return _read_json_array(p, "xz")
        case ".jsonl.xz":             return _read_jsonl(p, "xz")
        case _:                       raise ValueError(f"Unsupported table type: {suf}")

# ----------  Simple search logic  -------------------------------------------
class DatasetSearcher:
    def __init__(self, table: pd.DataFrame, text_cols: list[str] | None = None):
        self.df = table
        self.text_cols = text_cols or [c for c in table.columns if table[c].dtype == "object"]

    def query(self, pattern: str, regex: bool = False, case: bool = False,
              topk: int = 10) -> pd.DataFrame:
        """Return rows where *any* text column matches `pattern`."""
        pat = re.compile(pattern, 0 if case else re.I) if not regex else pattern
        mask = pd.Series(False, index=self.df.index)
        for col in self.text_cols:
            mask |= self.df[col].astype(str).str.contains(pat, regex=not isinstance(pat, re.Pattern))
        return self.df[mask].head(topk)
