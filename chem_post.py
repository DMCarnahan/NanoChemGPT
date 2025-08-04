import re, time
from typing import List, Dict

from rdkit import Chem
from rdkit.Chem.inchi import MolToInchiKey
import pubchempy as pcp

# ----------------------------------------------------------------------
# 1. canonicalise && valence check
# ----------------------------------------------------------------------
def _canonical_name(name: str) -> str:
    """
    Try to canonicalise a reagent/solvent name.
    • If it parses as SMILES, return its InChIKey.
    • Else return the PubChem standard name if available.
    • Fallback: original string.
    """
    # quick SMILES heuristic: contains [,=,# or digits
    if any(c in name for c in "[]=#0123456789"):
        mol = Chem.MolFromSmiles(name, sanitize=False)
        if mol is not None:
            try:
                Chem.SanitizeMol(mol)
                return MolToInchiKey(mol)
            except Exception:
                # invalid valence etc.
                return name

    # name → CID → canonical name
    try:
        cids = pcp.get_cids(name, 'name')
        if cids:
            comp = pcp.Compound.from_cid(cids[0])
            return comp.iupac_name.title() if comp.iupac_name else comp.synonyms[0]
    except Exception:
        pass
    return name


def _cas_number(name: str) -> str | None:
    """Fetch the first CAS RN from PubChem (fast local cache)."""
    try:
        cids = pcp.get_cids(name, 'name')
        if not cids:
            return None
        comp = pcp.Compound.from_cid(cids[0])
        for s in comp.synonyms:
            if re.match(r"^\d{2,7}-\d{2}-\d$", s):
                return s
    except Exception:
        return None
    return None


# ----------------------------------------------------------------------
# 2. split “add…, stir…” lines → two atomic steps
# ----------------------------------------------------------------------
_SPLIT_RX = re.compile(r"\b(?:then|and then|and|;)\s+", re.I)

def _split_compound_step(step: Dict) -> List[Dict]:
    """If the details text contains multiple verbs, split into sub-steps."""
    text = step["details"]
    parts = _split_rx.split(text)
    if len(parts) == 1:
        return [step]

    out = []
    for part in parts:
        part = part.strip().rstrip(".")
        if not part:
            continue
        # crude verb → action mapping
        if part.lower().startswith(("stir", "mix", "agitate")):
            act = step.copy(); act["action"] = "stir"; act["details"] = part
        elif part.lower().startswith(("heat", "reflux")):
            act = step.copy(); act["action"] = "heat"; act["details"] = part
        else:
            act = step.copy(); act["action"] = "add";  act["details"] = part
        out.append(act)
    return out


# ----------------------------------------------------------------------
# 3. main public helper
# ----------------------------------------------------------------------
def postprocess_steps(steps: List[Dict]) -> List[Dict]:
    """Enrich and clean the list returned by gpt_steps()."""
    final: List[Dict] = []

    for step in steps:
        # --- 3.1 split compound sentences
        for s in _split_compound_step(step):
            # --- 3.2 canonicalise reagents & solvents
            for field in ("reagents", "solvents"):
                if field in s:
                    canon = []
                    for r in s[field]:
                        r_can = _canonical_name(r)
                        cas   = _cas_number(r_can) or _cas_number(r)
                        if cas:
                            r_can = f"{r_can} (CAS {cas})"
                        canon.append(r_can)
                    s[field] = canon

            final.append(s)

            # polite delay for PubChem (≤ 5 req/s)
            time.sleep(0.05)
    return final
