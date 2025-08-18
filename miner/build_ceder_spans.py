from __future__ import annotations
import argparse, json, lzma, re, random, unicodedata, sys
from typing import List, Tuple, Dict, Any, Optional
from pathlib import Path
import spacy
from spacy.tokens import Doc, DocBin, Span
from runtime import material_ok

# -------------------- Configs --------------------

OPTYPE_TO_TRIGGERS = {
    "HeatingOperation": ["heat","heated","heating","maintain","maintained","anneal","annealed","reflux","calcine","calcined","sinter","sintered"],
    "StirringOperation": ["stir","stirred","stirring","agitate","mixed","mixing","mix"],
    "AdditionOperation": ["add","added","adding","introduce","charge","charged"],
    "InjectionOperation": ["inject","injected","injection"],
    "DegassingOperation": ["degas","degassed","purge","purged","sparge","sparged"],
    "CentrifugeOperation": ["centrifuge","centrifuged","centrifugation"],
    "WashingOperation": ["wash","washed","rinse","rinsed"],
    "DryingOperation": ["dry","dried","drying","evaporate","evaporated"],
    "FiltrationOperation": ["filter","filtered","filtration"],
    "SonicationOperation": ["sonicate","sonicated","sonication"],
    "_solidstate": ["grind","ground","mill","milled","press","pressed","pelletize","pelletized","calcine","calcined","sinter","sintered","anneal","annealed"]
}

VESSEL_WORDS = ["flask","beaker","vial","autoclave","round-bottom","tube","ampoule","crucible","furnace","reactor","tube furnace","muffle furnace"]
EQUIPMENT_WORDS = ["stir plate","hotplate","thermocouple","centrifuge","vacuum","oven","ultrasonicator","sonicator","balance","hood","ball mill","press"]
ATMOS_WORDS = ["argon","nitrogen","air","oxygen","vacuum","N2","Ar","O2","H2"]

# --- Units for AMOUNT pairing ONLY ---
VOLUME_UNITS = r"(?:μL|µL|uL|mL|L)"
MASS_UNITS   = r"(?:μg|µg|ug|mg|g|kg)"
MOL_UNITS    = r"(?:μmol|µmol|umol|mmol|mol)"
LENGTH_UNITS = r"(?:nm|μm|µm|mm|cm)"   # keep if your gold treats lengths as AMOUNT

# DO NOT include concentration or speed units here
UNIT_RX = rf"(?:{VOLUME_UNITS}|{MASS_UNITS}|{MOL_UNITS}|{LENGTH_UNITS})"

# --- Concentration & Speed handled separately ---
CONC_UNIT_RX  = r"(?:M|mM|μM|µM|uM|%|wt%|vol%|v/v|w/w|mol/?L|mol·L(?:-1|⁻¹))"
SPEED_UNIT_RX = r"(?:rpm|r[\.\s·]?min(?:-1|⁻¹|\^-1))"

NUM_RX = r"""
(?:
  (?:~|≈|about|ca\.?)\s*?
)?
(?:
  \d+(?:[.,]\d+)?(?:\s*[×x]\s*\d+(?:[.,]\d+)?)?(?:\s*(?:e|E)[±−-]?\d+)?   # 1.2e-3, 2x10
  |
  \d+(?:[.,]\d+)?\s*[–—-]\s*\d+(?:[.,]\d+)?                               # 10–12
  |
  \d+(?:[.,]\d+)?\s*[×x]\s*10(?:[-−^]?\d+|⁻\d+)                           # 1×10^-3, 1×10⁻3
)
""".strip()

TEMP_RX = r"(?<![\w])(-?\d+(?:[.,]\d+)?)\s*(?:°\s*)?(?:C|K|F)\b"
TIME_RX = r"(?<![\w])(\d+(?:[.,]\d+)?)\s*(?:s|sec|secs|second|seconds|min|mins|minute|minutes|h|hr|hrs|hour|hours)\b"
SPEED_RX = r"(?<![\w])(\d+(?:[.,]\d+)?)\s*(?:rpm)\b"
CONC_RX = r"(?<![\w])(\d+(?:[.,]\d+)?)\s*(?:M|mM|µM|uM|wt%|vol%|%)\b"
SOLVENT_WHITELIST = {
    "water","ethanol","methanol","isopropanol","isopropyl alcohol","ipa","propanol",
    "butanol","toluene","hexane","heptane","octane","chloroform","dichloromethane","dcm",
    "acetonitrile","acetone","dmf","dimethylformamide","dmso","dimethyl sulfoxide",
    "oleylamine","oleic acid","trioctylphosphine","top","tbhp","pvp","peg"
}
TEXT_KEYS = ["paragraph_string","paragraph","text","context","sentence","raw","span_text","source_text","proc_text","body","content","abstract"]
MATERIAL_KEY_HINTS = ["precursor","reactant","reagent","material","target","product","solvent","compound","salt","oxide"]

KEY_SCORE_HINTS = {"paragraph","operations","quantities","precursor","material","solvent","target"}

PAIR_RX = rf"(?P<num>{NUM_RX})\s*(?P<unit>{UNIT_RX})"
PAIR_RE = re.compile(PAIR_RX, re.I)
ACTION_FP_PATTERNS = [
    r"\bgrinding media\b", r"\bmilling (?:jar|jars)\b",
    r"\bpress (?:die|dies)\b", r"\bball mill\b"
]
# Heuristics to reduce MATERIAL over-labeling
TAIL_NONMAT = re.compile(
    r'\s*(?:'
    r'film(?:s)?|thin\s*film(?:s)?|coating(?:s)?|layer(?:s)?|'
    r'substrate(?:s)?|wafer(?:s)?|support(?:s)?|'
    r'powder(?:s)?|nanopowder(?:s)?|pellet(?:s)?|granule(?:s)?|grain(?:s)?|crystallite(?:s)?|'
    r'particle(?:s)?|nanoparticle(?:s)?|microparticle(?:s)?|'
    r'nanocrystal(?:s)?|nanocrystalline|nanorod(?:s)?|nanowire(?:s)?|nanosheet(?:s)?|nanofiber(?:s)?|nanoflake(?:s)?|nanotube(?:s)?|'
    r'sphere(?:s)?|hollow\s*sphere(?:s)?|core[-\s]*shell(?:s)?|yolk[-\s]*shell(?:s)?|'
    r'composite(?:s)?|nanocomposite(?:s)?|hybrid(?:s)?|heterostructure(?:s)?|'
    r'catalyst(?:s)?|photocatalyst(?:s)?|electrode(?:s)?|membrane(?:s)?|'
    r'sample(?:s)?|product(?:s)?|specimen(?:s)?|'
    r'solution(?:s)?|suspension(?:s)?|slurry|gel(?:s)?|sol(?:s)?'
    r')\s*$',
    re.I
)
DOPANT_RX = r"[A-Z][a-z]?(?:\d+|[0-9\-+½¼¾⁻⁺]*)"
COMPLEX_FORMULA_RX = rf"(?:{DOPANT_RX}(?:[:\-–—·•]{DOPANT_RX})+|[A-Z][a-z]?\([A-Za-z0-9\.\-+−⁻]+\)\d*(?:[A-Za-z0-9\.\-+−⁻]*)?)"
CHEMISH = re.compile(
    rf"(?:{COMPLEX_FORMULA_RX})|(?:[A-Z][a-z]?\d+)+|oxide|nitrate|acetate|chloride|sulfate|hydroxide|phosphate|carbonate|perovskite|aluminate",
    re.I
)

STOP_MATERIAL = {
    "ito glass","fto glass","glass slide","quartz tube","quartz boat","alumina crucible",
    "al2o3 substrate","sapphire substrate","sapphire wafer","filter paper","teflon liner",
    "alumina tube","mortar","pestle","ball mill","press die","polypropylene bottle"
}
STOP_PATTERNS = re.compile(
    r"\b(?:ito|fto)\s+glass\b|"
    r"\b(quartz|alumina|sapphire)\s+(tube|boat|crucible|substrate|wafer)\b|"
    r"\b(graphite\s+foil|nickel\s+foam)\b",
    re.I
)

# -------------------- Helpers --------------------
def looks_like_fp_action(text: str, s: int, e: int) -> bool:
    win = text[max(0, s-12): min(len(text), e+20)].lower()
    return any(re.search(p, win) for p in ACTION_FP_PATTERNS)

def norm_text(t: str) -> str:
    t = unicodedata.normalize("NFKC", t)
    t = t.replace("\u00b0", "°")
    t = re.sub(r"[ \t\r\f\v]+", " ", t)
    return t

def _json_open(path: Path):
    # open lzma .xz or plain .json
    if str(path).lower().endswith(".xz"):
        return lzma.open(path, "rt", encoding="utf-8")
    return open(path, "rt", encoding="utf-8")

def _iter_jsonl(f):
    for line in f:
        line = line.strip()
        if not line:
            continue
        yield json.loads(line)

def _nested_lists_of_dicts(obj: Any, path: str="") -> List[tuple]:
    """Return [(path, list_obj)] for any list whose elements are dicts."""
    out = []
    if isinstance(obj, list) and obj and isinstance(obj[0], dict):
        out.append((path, obj))
    elif isinstance(obj, dict):
        for k, v in obj.items():
            subpath = f"{path}.{k}" if path else str(k)
            out.extend(_nested_lists_of_dicts(v, subpath))
    return out

def _score_candidate_list(lst: list) -> int:
    # Heuristic: length + presence of indicative keys in sample dicts
    score = len(lst)
    if lst and isinstance(lst[0], dict):
        keys = set(lst[0].keys())
        score += 50 * len(keys & KEY_SCORE_HINTS)
    return score

def load_records(path: Path, root_key: Optional[str]=None) -> List[dict]:
    with _json_open(path) as f:
        head = f.read(4096)
        f.seek(0)
        # JSON array
        if head.lstrip().startswith("["):
            data = json.load(f)
            if not isinstance(data, list):
                raise ValueError("Expected list at top-level.")
            return data
        # Try full JSON
        try:
            obj = json.load(f)
        except json.JSONDecodeError:
            # treat as JSONL
            f.seek(0)
            return list(_iter_jsonl(f))

    if not isinstance(obj, dict):
        raise ValueError(f"Unsupported top-level type: {type(obj).__name__}")

    # If user specified a dot-path
    if root_key:
        cur = obj
        for part in root_key.split("."):
            if isinstance(cur, dict) and part in cur:
                cur = cur[part]
            else:
                raise ValueError(f"--root-key path '{root_key}' not found")
        if isinstance(cur, list) and (not cur or isinstance(cur[0], dict)):
            return cur
        raise ValueError(f"--root-key '{root_key}' doesn't resolve to list[dict]")

    # Auto-detect: find all nested lists of dicts, pick the best-scoring
    candidates = _nested_lists_of_dicts(obj)
    if not candidates:
        raise ValueError("No list[dict] candidates found in JSON object")
    best_path, best_list = max(candidates, key=lambda kv: _score_candidate_list(kv[1]))
    # print hint
    print(f"[info] Auto-selected root at '{best_path}' with {len(best_list)} records", file=sys.stderr)
    return best_list

def pick_text(rec: dict, forced_key: Optional[str]=None) -> str:
    if forced_key and isinstance(rec.get(forced_key), str):
        return norm_text(rec[forced_key])
    for k in TEXT_KEYS:
        v = rec.get(k)
        if isinstance(v, str) and len(v) >= 30:
            return norm_text(v)
    # scan nested dicts for a plausible text field
    for k, v in rec.items():
        if isinstance(v, dict):
            for kk in TEXT_KEYS:
                vv = v.get(kk)
                if isinstance(vv, str) and len(vv) >= 30:
                    return norm_text(vv)
    # longest string anywhere
    best = ""
    def scan(obj):
        nonlocal best
        if isinstance(obj, str) and len(obj) > len(best):
            best = obj
        elif isinstance(obj, dict):
            for vv in obj.values(): scan(vv)
        elif isinstance(obj, list):
            for it in obj: scan(it)
    scan(rec)
    return norm_text(best)

def find_all(text: str, sub: str) -> List[Tuple[int,int]]:
    spans = []
    sub = norm_text(sub).strip()
    if not sub: return spans
    lower = text.lower(); needle = sub.lower()
    start = 0
    while True:
        i = lower.find(needle, start)
        if i == -1: break
        spans.append((i, i+len(sub)))
        start = i + 1
    return spans

def regex_find(text: str, pattern: str):
    for m in re.finditer(pattern, text, flags=re.I):
        yield (m.start(), m.end(), m.group(0))

def choose_non_overlapping(spans):
    spans_sorted = sorted(spans, key=lambda x: (-(x[1]-x[0]), x[0]))
    chosen = []; used = []
    for s,e,lab,src in spans_sorted:
        if any(not (e <= us or s >= ue) for us,ue in used):
            continue
        chosen.append((s,e,lab,src)); used.append((s,e))
    return sorted(chosen, key=lambda x: x[0])

def collect_material_strings(rec: dict, extra_keys: List[str] | None = None) -> List[str]:
    """Collect candidate MATERIAL strings from the JSON record."""
    mats: List[str] = []
    keys_set = set(extra_keys or [])

    def maybe_add(val):
        if isinstance(val, str):
            s = norm_text(val).strip()
            if s:
                mats.append(s)

    def dig(obj, key_name: str = ""):
        if isinstance(obj, dict):
            for k, v in obj.items():
                lk = k.lower()
                if (k in keys_set) or any(h in lk for h in MATERIAL_KEY_HINTS):
                    if isinstance(v, list):
                        for it in v:
                            if isinstance(it, dict):
                                for subk in ("material_string","name","formula","chemical","label","material","compound"):
                                    if subk in it:
                                        maybe_add(it.get(subk))
                            else:
                                maybe_add(it)
                    elif isinstance(v, dict):
                        for subk in ("material_string","name","formula","chemical","label","material","compound"):
                            if subk in v:
                                maybe_add(v.get(subk))
                        dig(v, lk)
                    else:
                        maybe_add(v)
                else:
                    dig(v, lk)
        elif isinstance(obj, list):
            for it in obj:
                dig(it, key_name)
        # scalars: ignore

    dig(rec)

    out: List[str] = []
    seen = set()
    for mstr in mats:
        core = TAIL_NONMAT.sub("", mstr).strip()
        core = re.sub(r'^\s*(?:aqueous|ethanolic|methanolic)?\s*solution of\s+', '', core, flags=re.I)
        if not core:
            continue
        low = core.lower()
        if low in seen:
            continue
        # Always allow common solvents/reagents
        if low in SOLVENT_WHITELIST:
            out.append(core); seen.add(low); continue
        # Keep multi-word or anything that looks chemical-ish
        if (' ' in core) or CHEMISH.search(core):
            out.append(core); seen.add(low)
    return out[:100]

def collect_action_triggers(rec: dict) -> List[str]:
    triggers = set()
    ops = rec.get("operations")
    if isinstance(ops, list):
        for op in ops:
            if isinstance(op, dict):
                if isinstance(op.get("string"), str):
                    triggers.add(op["string"])
                t = op.get("type")
                if isinstance(t, str):
                    for trig in OPTYPE_TO_TRIGGERS.get(t, []):
                        triggers.add(trig)
    for t in OPTYPE_TO_TRIGGERS["_solidstate"]:
        triggers.add(t)
    return sorted(triggers)

def build_spans_from_record(text: str, rec: Dict[str, Any], extra_material_keys: List[str] | None = None) -> List[Tuple[int, int, str, str]]:
    text = norm_text(text)
    spans: List[Tuple[int, int, str, str]] = []
    covered: List[Tuple[int, int]] = []

    _action_words = collect_action_triggers(rec) or []
    ACTION_NEAR_RX = re.compile(
        rf"\b({'|'.join(map(re.escape, _action_words))})(?:ed|ing|s)?\b", re.I
    ) if _action_words else None

    # -------- MATERIAL (with atmosphere + stoplist + near-ACTION gates) --------
    for mstr in (collect_material_strings(rec, extra_material_keys) or []):
        core = mstr.strip()
        for a, b in find_all(text, mstr):
            if core.lower() in {"o2", "oxygen", "n2", "nitrogen", "air", "argon", "ar", "h2", "hydrogen"}:
                win_atm = text[max(0, a-25): b+25]
                if re.search(r"\b(flow|atmosphere|under|in|purge|gas)\b", win_atm, re.I):
                    continue

            low = core.lower()
            win = text[max(0, a-30): b+30]
            if low in STOP_MATERIAL or STOP_PATTERNS.search(win):
                continue

            if " " not in core and not CHEMISH.search(core) and ACTION_NEAR_RX:
                if not ACTION_NEAR_RX.search(win):
                    continue

            if not material_ok(core):
                continue

            spans.append((a, b, "MATERIAL", "materials"))



    # -------- AMOUNT+UNIT: pair-first extraction --------
    for m in PAIR_RE.finditer(text):
        a, b = m.start("num"), m.end("num")
        u1, u2 = m.start("unit"), m.end("unit")
        spans.append((a, b, "AMOUNT", "pair.num"))
        spans.append((u1, u2, "UNIT", "pair.unit"))
        covered.append((a, b))
        covered.append((u1, u2))

    def overlaps_covered(s: int, e: int) -> bool:
        return any(not (e <= cs or s >= ce) for cs, ce in covered)

    # -------- ACTION triggers (morphology & hyphen/space tolerant) --------
    def _build_action_pat(trig: str) -> re.Pattern:
        slug = re.escape(trig).replace(r"\ ", r"[-\s]*")  # allow hyphen or space
        return re.compile(rf"(?<![A-Za-z0-9]){slug}(?:ed|ing|s)?(?![A-Za-z0-9])", re.I)

    try:
        action_trigs = collect_action_triggers(rec) or []
    except Exception:
        action_trigs = []

    for trig in action_trigs:
        pat = _build_action_pat(trig)
        for m in pat.finditer(text):
            s, e = m.start(), m.end()
            try:
                if looks_like_fp_action(text, s, e):
                    continue
            except Exception:
                pass
            spans.append((s, e, "ACTION", "op.trigger"))

# -------- Structured quantities from JSON (also mark covered) --------
    qs = rec.get("quantities")
    if isinstance(qs, list):
        for q in qs:
            if isinstance(q, dict):
                for qd in q.get("quantity", []) or []:
                    if isinstance(qd, dict):
                        num = qd.get("number"); unit = qd.get("unit")
                        if num is None or not unit:
                            continue
                        num_s = str(num).replace(",", ".")
                        u = str(unit).strip()

                        # AMOUNT units (mass/vol/mol[/length])
                        if re.fullmatch(UNIT_RX, u, flags=re.I):
                            pat = rf"(?<!\d){re.escape(num_s)}\s*{re.escape(u)}"
                            for m in re.finditer(pat, text, flags=re.I):
                                ns, ne = m.start(), m.end()
                                us, ue = ne - len(u), ne
                                spans.append((ns, ne, "AMOUNT", "quantities"))
                                spans.append((ns, ns + len(num_s), "AMOUNT", "quantities.num"))
                                spans.append((us, ue, "UNIT", "quantities.unit"))
                                covered.append((ns, ns + len(num_s)))
                                covered.append((us, ue))

                        # Concentration units → CONC label
                        elif re.fullmatch(CONC_UNIT_RX, u, flags=re.I):
                            pat = rf"(?<!\d){re.escape(num_s)}\s*{re.escape(u)}"
                            for m in re.finditer(pat, text, flags=re.I):
                                cs, ce = m.start(), m.end()
                                spans.append((cs, ce, "CONC", "quantities.conc"))
                                covered.append((cs, ce))

                        # Speed units → SPEED label
                        elif re.fullmatch(SPEED_UNIT_RX, u, flags=re.I):
                            pat = rf"(?<!\d){re.escape(num_s)}\s*{re.escape(u)}"
                            for m in re.finditer(pat, text, flags=re.I):
                                ss, se = m.start(), m.end()
                                spans.append((ss, se, "SPEED", "quantities.speed"))
                                covered.append((ss, se))

                        # Unknown unit type → skip to avoid polluting AMOUNT
                        else:
                            continue

    # -------- Regex-based labels (skip if already covered) --------
    for s, e, _ in regex_find(text, TEMP_RX):
        spans.append((s, e, "TEMP", "regex"))
    for s, e, _ in regex_find(text, TIME_RX):
        spans.append((s, e, "TIME", "regex"))
    for s, e, _ in regex_find(text, SPEED_RX):
        if not overlaps_covered(s, e):
            spans.append((s, e, "SPEED", "regex"))
    for s, e, _ in regex_find(text, CONC_RX):
        if not overlaps_covered(s, e):
            spans.append((s, e, "CONC", "regex"))

    # -------- Lexicons --------
    for w in VESSEL_WORDS:
        for m in re.finditer(rf"\b{re.escape(w)}\b", text, flags=re.I):
            spans.append((m.start(), m.end(), "VESSEL", "lex.vessel"))
    for w in EQUIPMENT_WORDS:
        for m in re.finditer(rf"\b{re.escape(w)}\b", text, flags=re.I):
            spans.append((m.start(), m.end(), "EQUIPMENT", "lex.equipment"))
    for w in ATMOS_WORDS:
        for m in re.finditer(rf"\b{re.escape(w)}\b", text, flags=re.I):
            spans.append((m.start(), m.end(), "ATMOS", "lex.atmos"))

    return choose_non_overlapping(spans)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Path to dataset .json[.xz] (list/dict/JSONL)")
    ap.add_argument("--outdir", required=True, help="Output directory for .spacy files")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--dev_frac", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--text-key", type=str, default=None)
    ap.add_argument("--material-keys", nargs="*", default=None)
    ap.add_argument("--root-key", type=str, default=None, help="Dot path inside JSON object to list[dict]")
    args = ap.parse_args()

    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)
    data_path = Path(args.input)

    records = load_records(data_path, root_key=args.root_key)
    if args.limit:
        records = records[: args.limit]

    nlp = spacy.blank("en"); nlp.add_pipe("sentencizer")
    db_train = DocBin(store_user_data=False); db_dev = DocBin(store_user_data=False)

    rng = random.Random(args.seed)
    idxs = list(range(len(records))); rng.shuffle(idxs)
    split = int(len(idxs) * (1.0 - args.dev_frac))
    train_idxs = set(idxs[:split])

    n_train = n_dev = 0
    for rec_idx in idxs:
        rec = records[rec_idx]
        text = pick_text(rec, forced_key=args.text_key)
        if not text.strip():
            continue
        spans = build_spans_from_record(text, rec, extra_material_keys=args.material_keys)
        doc = nlp.make_doc(text)
        ents = []
        for s,e,label,src in spans:
            span = doc.char_span(s, e, label=label, alignment_mode="contract")
            if span is not None and span.text.strip() and len(span)>0:
                ents.append(span)
        ents = spacy.util.filter_spans(ents)
        doc.ents = ents
        if rec_idx in train_idxs:
            db_train.add(doc); n_train += 1
        else:
            db_dev.add(doc); n_dev += 1

    (outdir / "train.spacy").write_bytes(db_train.to_bytes())
    (outdir / "dev.spacy").write_bytes(db_dev.to_bytes())
    with open(outdir / "labels.json", "w", encoding="utf-8") as f:
        json.dump(["ACTION","MATERIAL","AMOUNT","UNIT","TEMP","TIME","SPEED","CONC","VESSEL","ATMOS","EQUIPMENT"], f, indent=2)

    print(f"Wrote {n_train} train docs → {outdir/'train.spacy'}")
    print(f"Wrote {n_dev} dev docs   → {outdir/'dev.spacy'}")
    print("Tip: Use --root-key if auto-selection picks the wrong list; run peek_dataset.py to see candidates.", file=sys.stderr)

if __name__ == "__main__":
    main()
