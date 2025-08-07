import json, pathlib, gzip, lzma, pandas as pd

def _load_csv(p):           return pd.read_csv(p).to_dict(orient="records")
def _load_jsonl(p):
    with gzip.open(p, "rt") if p.suffix == ".gz" else p.open() as f:
        return [json.loads(line) for line in f if line.strip()]

def _load_json_array(p):
    with (lzma.open(p) if p.suffix == ".xz" else p.open()) as f:
        data = json.load(f)
    return data if isinstance(data, list) else data.get("reactions", [])

LOADERS = {".csv":_load_csv, ".csv.gz":_load_csv,
           ".jsonl":_load_jsonl, ".jsonl.gz":_load_jsonl,
           ".json":_load_json_array, ".json.xz":_load_json_array}

def load_records(path):
    p = pathlib.Path(path)
    for suff, fn in LOADERS.items():
        if p.name.endswith(suff):
            return fn(p)
    raise ValueError(f"Unsupported file type {p}")
