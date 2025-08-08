from __future__ import annotations
import os, re, backoff, requests
from typing import Iterable, Dict, Any, List, Optional
from rapidfuzz import process, fuzz

try:
    import pubchempy as pcp
except Exception:
    pcp = None

# --- PubChem helpers ---------------------------------------------------------

@backoff.on_exception(backoff.expo, (requests.exceptions.RequestException, pcp.PubChemHTTPError if pcp else Exception), max_time=20)
def _pcp_compounds(q: str) -> List["pcp.Compound"]:
    if not pcp:
        return []
    # try name → fallback to synonym search
    try:
        hits = pcp.get_compounds(q, "name")
        if hits: return hits
    except Exception:
        pass
    try:
        return pcp.get_compounds(q, "synonym")
    except Exception:
        return []

def _best_hit(name: str, cands: List["pcp.Compound"]) -> Optional["pcp.Compound"]:
    if not cands: return None
    # score by fuzzy match on IUPAC/Title/Synonyms
    scored = []
    for c in cands:
        s = [getattr(c, "iupac_name", "") or "", getattr(c, "title", "") or ""]
        try:
            s += (c.synonyms or [])
        except Exception:
            pass
        best = process.extractOne(name, s, scorer=fuzz.WRatio)
        scored.append((best[1] if best else 0, c))
    scored.sort(reverse=True, key=lambda t: t[0])
    return scored[0][1]

def _extract_cas(c: "pcp.Compound") -> Optional[str]:
    try:
        syns = c.synonyms or []
    except Exception:
        syns = []
    for s in syns:
        if re.fullmatch(r"\d{2,7}-\d{2}-\d", s):  # CAS-ish
            return s
    return None

def enrich_materials(lines: Iterable[str]) -> List[Dict[str, Any]]:
    """Turn bullet lines into structured materials with PubChem metadata."""
    out: List[Dict[str, Any]] = []
    for raw in lines:
        name = re.sub(r"^[\-\*\d\.\)\s]+", "", raw).strip()
        rec: Dict[str, Any] = {"name": name}
        if pcp:
            try:
                hits = _pcp_compounds(name)
                best = _best_hit(name, hits)
                if best:
                    rec.update({
                        "cid": best.cid,
                        "iupac_name": getattr(best, "iupac_name", None),
                        "canonical_smiles": getattr(best, "canonical_smiles", None),
                        "molecular_formula": getattr(best, "molecular_formula", None),
                        "cas": _extract_cas(best),
                    })
            except Exception as e:
                rec.setdefault("notes", f"pubchem: {e}")
        out.append(rec)
    return out
