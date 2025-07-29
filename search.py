from __future__ import annotations
import os, re, json, time, hashlib, pathlib
from typing import List, Dict, Any, Tuple, Optional
import requests

CROSSREF_URL = "https://api.crossref.org/works"
ARXIV_URL    = "http://export.arxiv.org/api/query"

CACHE_DIR     = os.getenv("SEARCH_CACHE_DIR", "/tmp/search_cache")
CACHE_TTL_SEC = int(os.getenv("SEARCH_CACHE_TTL", "86400"))
SEARCH_EXPAND = os.getenv("SEARCH_EXPAND", "1") == "1"
STRICT_TITLE_MATERIAL = os.getenv("STRICT_TITLE_MATERIAL", "1") == "1"
STRICT_SIZE   = os.getenv("STRICT_SIZE", "0") == "1"
SIZE_TOL_FRAC = float(os.getenv("SIZE_TOL_FRAC", "0.2"))

DEFAULT_JOURNAL_BOOSTS = {
    "ACS Nano": 2.0, "Nano Letters": 2.0, "Advanced Materials": 2.0,
    "Chem. Mater": 1.5, "Chemistry of Materials": 1.5, "J. Phys. Chem. C": 1.5,
    "Nanoscale": 1.2, "Langmuir": 1.2, "Small": 1.2, "CrystEngComm": 1.0,
}
try:
    JOURNAL_BOOSTS = json.loads(os.getenv("JOURNAL_WHITELIST", "")) or DEFAULT_JOURNAL_BOOSTS
except Exception:
    JOURNAL_BOOSTS = DEFAULT_JOURNAL_BOOSTS

MATERIAL_SYNONYMS = {
    "iron oxide": ["iron oxide", "magnetite", "fe3o4", "maghemite", "fe2o3"],
    "zinc oxide": ["zno", "zinc oxide"],
    "titanium dioxide": ["tio2", "titanium dioxide", "anatase", "rutile"],
    "silica": ["silica", "sio2", "stöber", "stober"],
    "nickel oxide": ["nio", "nickel oxide"],
    "cobalt oxide": ["coo", "co3o4", "cobalt oxide"],
    "gold": ["gold", "au"],
    "silver": ["silver", "ag"],
    "cadmium selenide": ["cdse", "cadmium selenide"],
    "lead sulfide": ["pbs", "lead sulfide"],
    "perovskite": ["perovskite", "mapbi3", "fapbi3", "cspbbr3", "cspbi3"],
    "graphene": ["graphene", "graphite oxide", "go", "rgo"],
}
SHAPE_SYNONYMS = {
    "sphere": ["nanosphere", "nanospheres", "spherical"],
    "rod": ["nanorod", "nanorods", "rod-like"],
    "wire": ["nanowire", "nanowires"],
    "cube": ["nanocube", "nanocubes", "cubic"],
    "sheet": ["nanosheet", "nanosheets", "2d sheet"],
    "tube": ["nanotube", "nanotubes"],
}
METHOD_TERMS = ["co-precipitation","coprecipitation","solvothermal","hydrothermal","hot injection","polyol","thermal decomposition","microemulsion","reverse micelle","seeded growth","stober","stöber","sol-gel"]
LIGAND_TERMS = ["oleylamine","oleic acid","pvp","ctab","sds","toab","topo","trioctylphosphine","citric acid","pei","dopamine"]
SIZE_RX = re.compile(r"\b(\d{1,3})\s*nm\b", re.I)

def _cache_key(url: str, params: Dict[str, Any]) -> str:
    s = url + "?" + "&".join(f"{k}={params[k]}" for k in sorted(params))
    import hashlib; return hashlib.sha1(s.encode("utf-8")).hexdigest()

def _cache_path(key: str) -> pathlib.Path:
    root = pathlib.Path(CACHE_DIR); root.mkdir(parents=True, exist_ok=True)
    return root / f"{key}.json"

def _cache_get(url: str, params: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    key = _cache_key(url, params); p = _cache_path(key)
    if p.exists():
        try:
            if CACHE_TTL_SEC > 0 and time.time() - p.stat().st_mtime <= CACHE_TTL_SEC:
                return json.loads(p.read_text("utf-8"))
        except Exception:
            return None
    return None

def _cache_set(url: str, params: Dict[str, Any], data: Dict[str, Any]) -> None:
    key = _cache_key(url, params); p = _cache_path(key)
    try: p.write_text(json.dumps(data), encoding="utf-8")
    except Exception: pass

def parse_query_features(q: str) -> dict:
    q_low = q.lower()
    mats, shapes, methods, ligands = [], [], [], []
    for canon, syns in MATERIAL_SYNONYMS.items():
        if any(s in q_low for s in syns + [canon]): mats.append(canon)
    for canon, syns in SHAPE_SYNONYMS.items():
        if any(s in q_low for s in syns + [canon]): shapes.append(canon)
    for t in METHOD_TERMS:
        if t in q_low: methods.append(t)
    for t in LIGAND_TERMS:
        if t in q_low: ligands.append(t)
    sizes = [int(m.group(1)) for m in SIZE_RX.finditer(q)]
    return {"materials": sorted(set(mats)),
            "shapes": sorted(set(shapes)),
            "sizes_nm": sizes,
            "methods": sorted(set(methods)),
            "ligands": sorted(set(ligands))}

def _or_group(tokens: List[str]) -> str:
    toks = [t for t in sorted(set(tokens)) if t]
    toks = [f'"{t}"' if " " in t and not t.startswith('"') else t for t in toks]
    return "(" + " OR ".join(toks) + ")" if toks else ""

def build_expanded_query(original_q: str, feats: dict) -> str:
    parts = [original_q.strip()]
    if feats["materials"]:
        mat_tokens = []
        for m in feats["materials"]:
            mat_tokens += MATERIAL_SYNONYMS.get(m, [m])
        parts.append(_or_group(mat_tokens))
    if feats["shapes"]:
        shape_tokens = []
        for s in feats["shapes"]:
            shape_tokens += SHAPE_SYNONYMS.get(s, [s])
        parts.append(_or_group(shape_tokens))
    if feats["methods"]: parts.append(_or_group(feats["methods"]))
    if feats["ligands"]: parts.append(_or_group(feats["ligands"]))
    return " ".join(p for p in parts if p).strip() or original_q

def _norm_crossref_item(x: Dict[str, Any]) -> Dict[str, Any]:
    doi = (x.get("DOI") or "").lower()
    title = " ".join(x.get("title") or [])[:500]
    yr = None
    try: yr = int((x.get("issued", {}).get("date-parts") or [[None]])[0][0])
    except Exception: pass
    url = x.get("URL") or (f"https://doi.org/{doi}" if doi else None)
    venue = ""
    if x.get("container-title"): venue = x.get("container-title", [""])[0] or ""
    publisher = x.get("publisher") or ""
    venue_str = " • ".join(v for v in [venue, publisher] if v)
    return {"title": title, "year": yr, "venue": venue_str, "source": "Crossref", "doi": doi or None, "url": url, "abstract": ""}

def _request_with_cache(url: str, params: Dict[str, Any]) -> Dict[str, Any]:
    cached = _cache_get(url, params)
    if cached is not None: return cached
    r = requests.get(url, params=params, timeout=25)
    r.raise_for_status()
    data = r.json() if "crossref" in url else {"text": r.text}
    _cache_set(url, params, data)
    return data

def _search_crossref(q: str, rows: int = 25) -> List[Dict[str, Any]]:
    params = {"query": q, "filter": "type:journal-article,from-pub-date:2005-01-01", "rows": rows, "select": "title,DOI,issued,container-title,publisher,URL", "sort": "relevance"}
    data = _request_with_cache(CROSSREF_URL, params)
    items = (data.get("message", {}).get("items") or [])
    return [_norm_crossref_item(x) for x in items]

def _search_arxiv(q: str, rows: int = 16) -> List[Dict[str, Any]]:
    a_q = f'all:"{q}"'
    params = {"search_query": a_q, "start": 0, "max_results": rows, "sortBy": "relevance"}
    data = _request_with_cache(ARXIV_URL, params)
    txt = data.get("text", "")
    out = []
    for entry in txt.split("<entry>")[1:]:
        def _g(tag):
            s = f"<{tag}>"; e = f"</{tag}>"
            return entry.split(s,1)[1].split(e,1)[0] if s in entry and e in entry else ""
        title = re.sub(r"\s+", " ", _g("title")).strip()
        link  = ""
        for p in entry.split("<link"):
            if 'rel="alternate"' in p and 'href="' in p:
                link = p.split('href="',1)[1].split('"',1)[0]
        summary = re.sub(r"\s+", " ", _g("summary")).strip()
        yearm  = re.search(r"<published>(\d{4})-", entry)
        yr     = int(yearm.group(1)) if yearm else None
        out.append({"title": title, "year": yr, "venue": "arXiv", "source": "arXiv", "doi": None, "url": link or None, "abstract": summary})
    return out

def _journal_boost(venue: str) -> float:
    v = venue or ""
    boost = 0.0
    for name, w in JOURNAL_BOOSTS.items():
        if name.lower() in v.lower(): boost += float(w)
    return boost

def _score_item(q: str, feats: dict, it: Dict[str, Any]) -> float:
    title = (it.get("title") or "").lower()
    abstr = (it.get("abstract") or "").lower()
    venue = (it.get("venue") or "")
    text  = f"{title}. {abstr}"
    score = 0.0

    if feats["materials"] and STRICT_TITLE_MATERIAL:
        if not any(tok in title for m in feats["materials"] for tok in MATERIAL_SYNONYMS.get(m, [m])):
            return -1e9

    for m in feats["materials"]:
        for tok in MATERIAL_SYNONYMS.get(m, [m]):
            if tok in title: score += 3
            elif tok in abstr: score += 1.5

    for s in feats["shapes"]:
        for tok in SHAPE_SYNONYMS.get(s, [s]):
            if tok in title: score += 2.5
            elif tok in abstr: score += 1.0

    found = [int(m.group(1)) for m in SIZE_RX.finditer(text)]
    if feats["sizes_nm"]:
        if STRICT_SIZE and not found: return -1e9
        for target in feats["sizes_nm"]:
            best = 0.0
            for nm in found:
                rel_err = abs(nm - target) / max(10, target)
                if rel_err <= SIZE_TOL_FRAC:
                    best = max(best, 2.0 + (1.0 if nm == target else 0.0))
            score += best

    for t in feats["methods"]:
        if t in title: score += 1.5
        elif t in abstr: score += 0.8
    for t in feats["ligands"]:
        if t in text: score += 0.5

    score += _journal_boost(venue)

    year = it.get("year")
    if isinstance(year, int):
        score += min(3.0, max(0.0, 0.1 * (year - 2010)))

    return score

def basic_search(q: str, n: int = 6) -> List[Dict[str, Any]]:
    feats = parse_query_features(q)
    qx = build_expanded_query(q, feats) if SEARCH_EXPAND else q

    items: List[Dict[str, Any]] = []
    try: items.extend(_search_crossref(qx, rows=max(24, n*4)))
    except Exception: pass
    try: items.extend(_search_arxiv(qx, rows=16))
    except Exception: pass

    scored = []
    for it in items:
        s = _score_item(q, feats, it)
        if s <= -1e8: continue
        scored.append((s, it))
    scored.sort(key=lambda t: (t[0], t[1].get("year") or 0), reverse=True)
    results = [it for _, it in scored[: n * 2]]
    return results[:n]

if __name__ == "__main__":
    import sys
    query = " ".join(sys.argv[1:]) or "50 nm iron oxide nanospheres coprecipitation"
    print(json.dumps(basic_search(query, n=6), indent=2))