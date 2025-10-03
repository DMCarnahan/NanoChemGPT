from __future__ import annotations
import os, re, joblib, importlib, pkgutil
from functools import lru_cache
from typing import Any, Dict, List

# ---- features used by material filter ----
CHEMISH = re.compile(
    r'(?:[A-Z][a-z]?\d+)+|oxide|nitrate|acetate|chloride|sulfate|hydroxide|'
    r'phosphate|carbonate|perovskite|aluminate',
    re.I
)
def feats(s: str) -> Dict[str, Any]:
    t = s.strip(); tl = t.lower()
    return {
        "len": len(t),
        "has_digit": any(c.isdigit() for c in t),
        "caps_ratio": (sum(c.isupper() for c in t) / max(1, len(t))),
        "chemish": int(bool(CHEMISH.search(t))),
        "ends_tail": int(bool(re.search(r'(film|powder|slurry|solution|composite|substrate|support)s?$', tl))),
        **{f"c3={tl[i:i+3]}": 1 for i in range(len(tl)-2)}
    }

import logging

try:
    from .basic_miner import BasicMiner
except Exception as e:
    logging.getLogger(__name__).error("Cannot import BasicMiner from .basic_miner: %s", e)
    raise

def get_miner(nlp_model: str | None = None, **kwargs):
    """
    Return a BasicMiner instance. Resolve the spaCy model path according to this priority:
      1. explicit nlp_model argument if provided and not empty
      2. SPACY_MODEL environment variable if set
      3. builtin trained model at harvester/miner/ner_model/model-best (relative to package)
      4. fall back to spaCy small english model
    """
    import os
    from pathlib import Path

    cand = nlp_model or os.getenv('SPACY_MODEL')
    if not cand:
        # Prefer the packaged 'model-best' shipped under harvester/miner/ner_model/model-best
        pkg_dir = Path(__file__).resolve().parent
        model_best = pkg_dir / 'ner_model' / 'model-best'
        if model_best.exists():
            cand = str(model_best)
    else:
        # If a candidate path/name was provided (env var or arg) and it points to an
        # existing filesystem path, prefer its resolved absolute path. This makes the
        # runtime-recorded requested path deterministic (absolute) for tests and
        # callers that inspect miner._model_path_requested.
        try:
            cand_path = Path(cand)
            if cand_path.exists():
                cand = str(cand_path.resolve())
        except Exception:
            # If Path() fails for whatever reason, keep the original candidate
            pass
    miner = BasicMiner(nlp_model=cand, **kwargs)
    # record which model path/name was requested so callers can inspect it
    try:
        miner._model_path_requested = cand
    except Exception:
        pass
    return miner

@lru_cache(maxsize=1)
def get_material_filter():
    path = os.path.join(os.path.dirname(__file__), "material_filter.joblib")
    if not os.path.exists(path):
        return None
    try:
        return joblib.load(path)  # {"vec": DictVectorizer, "clf": LogisticRegression}
    except Exception as e:
        print("[material_filter] load failed:", e)
        return None

def material_ok(text: str, thresh: float = 0.65) -> bool:
    mf = get_material_filter()
    if mf is None:
        return True
    v = mf["vec"].transform([feats(text)])
    p = mf["clf"].predict_proba(v)[0, 1]
    return p >= thresh

def _filter_materials_in_ops(ops: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    for op in ops or []:
        mats = op.get("materials")
        if not mats:
            continue
        kept = []
        for m in mats:
            txt = m if isinstance(m, str) else (m.get("text") or m.get("name") or m.get("label") or "")
            if not txt or material_ok(txt):
                kept.append(m)
        op["materials"] = kept
    return ops

def extract_procedure(text: str) -> dict:
    miner = get_miner()
    ops = miner.extract(text)
    ops = _filter_materials_in_ops(ops)
    expanded = miner.expand(ops)
    return {"operations": ops, "expanded": expanded}
