import json
import pathlib

p = pathlib.Path(r".\out\bundle.jsonl")  # change if needed
rows = 0
nonempty = 0
sample = None
with p.open("r", encoding="utf-8") as f:
    for line in f:
        rec = json.loads(line)
        paras = (rec.get("extractions", {}) or {}).get("methods_paragraphs", [])
        rows += len(paras)
        for pr in paras:
            t = (pr.get("text") or "").strip()
            if t:
                nonempty += 1
                if sample is None:
                    sample = t[:300]
print("paragraphs total:", rows, " nonempty:", nonempty)
print("sample:", sample)
