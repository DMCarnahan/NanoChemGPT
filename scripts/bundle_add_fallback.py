import json, re, sys, pathlib
from typing import List, Dict, Any

# --- simple signals ---
ACTION_WORDS = [
    "synthesize","synthesise","prepare","mix","stir","disperse","dissolve","add",
    "deposit","coat","spin coat","spin-coat","drop cast","drop-cast","anneal",
    "calcine","dry","age","wash","rinse","centrifuge","filter","grind",
    "heat","cool","reflux","sonicate","autoclave","hydrothermal","solvothermal","sol-gel"
]
NUM_RX    = r"(?:\d+(?:\.\d+)?(?:[Ee][+-]?\d+)?|\d{1,3}(?:,\d{3})+)"
VOLUME_U  = r"(?:μL|µL|uL|mL|L)"
MASS_U    = r"(?:μg|µg|ug|mg|g|kg)"
MOL_U     = r"(?:μmol|µmol|umol|mmol|mol)"
LENGTH_U  = r"(?:nm|μm|µm|mm|cm)"
UNIT_RX   = rf"(?:{VOLUME_U}|{MASS_U}|{MOL_U}|{LENGTH_U})"
TEMP_RX   = r"(?:\b\d{1,3}\s?°\s?[CF]\b|\b\d+(?:\.\d+)?\s?(?:K|°C|°F)\b)"
TIME_RX   = r"(?:\b\d+(?:\.\d+)?\s?(?:s|sec|secs|seconds|min|mins|h|hr|hrs|hours|day|days)\b)"
SPEED_RX  = r"(?:\b\d+(?:\.\d+)?\s?(?:rpm|r\.?\s?min(?:-1|⁻¹)?)\b)"
CONC_RX   = r"(?:\b\d+(?:\.\d+)?\s?(?:M|mM|μM|µM|uM|%|wt%|vol%|v/v|w/w)\b)"

ACTION_RE = re.compile(r"\b(" + "|".join(map(re.escape, ACTION_WORDS)) + r")(?:ed|ing|s)?\b", re.I)
NUMUNIT_RE= re.compile(rf"{NUM_RX}\s*{UNIT_RX}", re.I)
TEMP_RE   = re.compile(TEMP_RX, re.I)
TIME_RE   = re.compile(TIME_RX, re.I)
SPEED_RE  = re.compile(SPEED_RX, re.I)
CONC_RE   = re.compile(CONC_RX, re.I)

def split_paragraphs(section_text: str) -> List[str]:
    if not section_text: return []
    paras = [p.strip() for p in re.split(r"\n\s*\n", section_text) if p.strip()]
    if paras: return paras
    sents = re.split(r"(?<=[\.\?\!])\s+(?=[A-Z(])", section_text)
    chunk, buf = [], ""
    for s in sents:
        if len(buf) + len(s) < 600:
            buf = (buf + " " + s).strip()
        else:
            if buf: chunk.append(buf)
            buf = s
    if buf: chunk.append(buf)
    return chunk

def score_paragraph(p: str) -> float:
    if not p or len(p) < 80: return 0.0
    s  = 2.0 * len(ACTION_RE.findall(p))
    s += 1.5 * len(NUMUNIT_RE.findall(p))
    s += 1.0 * len(TEMP_RE.findall(p))
    s += 0.8 * len(TIME_RE.findall(p))
    s += 0.8 * len(SPEED_RE.findall(p))
    s += 0.8 * len(CONC_RE.findall(p))
    if re.search(r"^(To\s+|\bwas\b\s+(?:prepared|synthesized|deposited))", p, re.I): s += 0.8
    return s

def fallback_methods_from_sections(sections: List[Dict[str,Any]], top_k=5, min_score=1.6):
    cands = []
    for sec in sections or []:
        for para in split_paragraphs(sec.get("text","")):
            sc = score_paragraph(para)
            if sc >= min_score:
                cands.append((sc, {"heading": sec.get("heading",""), "text": para}))
    cands.sort(key=lambda x: x[0], reverse=True)
    return [d for _, d in cands[:top_k]]

try:
    from harvester.miner.runtime import extract_procedure
except Exception:
    extract_procedure = None

def main(in_path: str, out_path: str):
    src = pathlib.Path(in_path)
    dst = pathlib.Path(out_path)
    n_in = n_out = added = 0
    with src.open("r", encoding="utf-8") as fin, dst.open("w", encoding="utf-8") as fout:
        for line in fin:
            n_in += 1
            rec = json.loads(line)
            methods = (rec.get("extractions",{}) or {}).get("methods_paragraphs", [])
            if methods:
                fout.write(line); n_out += 1; continue
            sections = rec.get("sections", [])
            picks = fallback_methods_from_sections(sections, top_k=5, min_score=1.6)
            mp = []
            for d in picks:
                txt = d.get("text","")
                if not txt: continue
                if extract_procedure is not None:
                    ann = extract_procedure(txt)
                    mp.append({"text": txt, "operations": ann.get("operations", []), "expanded": ann.get("expanded", [])})
                else:
                    mp.append({"text": txt, "operations": [], "expanded": []})
            if "extractions" not in rec: rec["extractions"] = {}
            rec["extractions"]["methods_paragraphs"] = mp
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
            n_out += 1; added += len(mp)
    print(f"Processed {n_in} records → {n_out}. Added methods paragraphs to {added} entries.")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python scripts\\bundle_add_fallback.py <in_bundle.jsonl> <out_bundle.jsonl>")
        sys.exit(1)
    main(sys.argv[1], sys.argv[2])
