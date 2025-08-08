from __future__ import annotations
import os, re, time
from typing import List, Dict, Optional

try:
    import pubchempy as pcp
except Exception:
    pcp = None

# ----------------------------------------------------------------------
# 1. canonicalise && CAS
# ----------------------------------------------------------------------
def _smiles_to_inchikey(smiles: str) -> Optional[str]:
    if not Chem or not MolToInchiKey:
        return None
    try:
        mol = Chem.MolFromSmiles(smiles, sanitize=False)
        if mol is None:
            return None
        Chem.SanitizeMol(mol)  # will raise on bad valence
        return MolToInchiKey(mol)
    except Exception:
        return None

def _pubchem_standard_name(name: str) -> Optional[str]:
    """Return IUPAC or first synonym from PubChem."""
    if not pcp:
        return None
    try:
        cids = pcp.get_cids(name, "name")
        if not cids:
            # try synonym search as fallback
            cids = pcp.get_cids(name, "synonym")
        if not cids:
            return None
        comp = pcp.Compound.from_cid(cids[0])
        if getattr(comp, "iupac_name", None):
            return comp.iupac_name
        syns = getattr(comp, "synonyms", None) or []
        return syns[0] if syns else None
    except Exception:
        return None

def _cas_number(name_or_iupac: str) -> Optional[str]:
    """Fetch first CAS RN from PubChem synonyms."""
    if not pcp:
        return None
    try:
        cids = pcp.get_cids(name_or_iupac, "name")
        if not cids:
            return None
        comp = pcp.Compound.from_cid(cids[0])
        for s in getattr(comp, "synonyms", []) or []:
            if re.fullmatch(r"\d{2,7}-\d{2}-\d", s):
                return s
    except Exception:
        return None
    return None

def _canonical_name(name: str) -> str:
    """
    Canonicalize a reagent/solvent name.
    - If it looks like SMILES and RDKit is present, return InChIKey.
    - Else use PubChem IUPAC/first synonym if available.
    - Fallback: original string.
    """
    looks_like_smiles = any(c in name for c in "[]+=#0123456789") and " " not in name
    if looks_like_smiles:
        ik = _smiles_to_inchikey(name)
        if ik:
            return ik
    std = _pubchem_standard_name(name)
    return std or name

# ----------------------------------------------------------------------
# 2. split “add…, stir…” lines → two atomic steps
# ----------------------------------------------------------------------
_SPLIT_RX = re.compile(r"\b(?:then|and then|and|;)\s+", re.I)

def _split_compound_step(step: Dict) -> List[Dict]:
    text = step.get("details", "") or ""
    parts = _SPLIT_RX.split(text)
    if len(parts) == 1:
        return [step]

    out = []
    for part in parts:
        part = part.strip().rstrip(".")
        if not part:
            continue
        lower = part.lower()
        act = dict(step)  # shallow copy
        if lower.startswith(("stir", "mix", "agitate")):
            act["action"] = "stir"
        elif lower.startswith(("heat", "reflux")):
            act["action"] = "heat"
        elif lower.startswith(("cool", "ice-bath", "ice bath", "quench")):
            act["action"] = "cool"
        else:
            act["action"] = act.get("action") or "add"
        act["details"] = part
        out.append(act)
    return out

# ----------------------------------------------------------------------
# 3. main public helper
# ----------------------------------------------------------------------
def postprocess_steps(steps: List[Dict]) -> List[Dict]:
    """
    Enrich and clean the list returned by gpt_steps().
    - Split multi-verb sentences.
    - Canonicalize reagent/solvent names.
    - Add CAS when available.
    """
    final: List[Dict] = []

    for step in steps or []:
        for s in _split_compound_step(step):
            # canonicalise reagents & solvents
            for field in ("reagents", "solvents"):
                vals = s.get(field) or []
                canon = []
                for r in vals:
                    r_can = _canonical_name(r)
                    cas = _cas_number(r_can) or _cas_number(r)
                    if cas:
                        r_can = f"{r_can} (CAS {cas})"
                    canon.append(r_can)
                if canon:
                    s[field] = canon
            final.append(s)
            # polite PubChem pacing (<= ~5 req/s)
            time.sleep(0.05)

    return final
