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

def _locate_BasicMiner():
    """Find and return the BasicMiner class inside the miner package."""
    # common filenames to try first
    candidates = ["miner.basic_miner", "miner.BasicMiner", "miner.basic", "miner.miner"]
    for mn in candidates:
        try:
            mod = importlib.import_module(mn)
            cls = getattr(mod, "BasicMiner", None)
            if cls:
                return cls
        except ModuleNotFoundError:
            continue
        except Exception:
            continue
    # last resort: scan all non-package modules in this package
    pkgdir = os.path.dirname(__file__)
    for _, name, ispkg in pkgutil.iter_modules([pkgdir]):
        if ispkg:
            continue
        try:
            mod = importlib.import_module(f"{__package__}.{name}")
            cls = getattr(mod, "BasicMiner", None)
            if cls:
                return cls
        except Exception:
            pass
    raise ImportError(
        "Could not locate class 'BasicMiner' in the 'miner' package. "
        "Put it in miner/basic_miner.py (class BasicMiner) or adjust _locate_BasicMiner()."
    )

@lru_cache(maxsize=1)
def get_miner():
    BM = _locate_BasicMiner()
    return BM()

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
