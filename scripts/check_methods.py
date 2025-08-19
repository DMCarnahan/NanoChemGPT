import json, sys, pathlib
p = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else r".\out\bundle.jsonl")
papers = nonempty = 0
with p.open("r", encoding="utf-8") as f:
    for line in f:
        rec = json.loads(line)
        papers += 1
        paras = (rec.get("extractions", {}) or {}).get("methods_paragraphs", [])
        nonempty += sum(1 for pr in paras if (pr.get("text") or "").strip())
print("papers:", papers, "nonempty methods paragraphs:", nonempty)