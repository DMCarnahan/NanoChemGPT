"""Keep retrieved evidence scoped to the material in the current request."""

from __future__ import annotations

import re
import unicodedata

_ELEMENTS = set(
    "H He Li Be B C N O F Ne Na Mg Al Si P S Cl Ar K Ca Sc Ti V Cr Mn Fe Co Ni "
    "Cu Zn Ga Ge As Se Br Kr Rb Sr Y Zr Nb Mo Tc Ru Rh Pd Ag Cd In Sn Sb Te I Xe "
    "Cs Ba La Ce Pr Nd Pm Sm Eu Gd Tb Dy Ho Er Tm Yb Lu Hf Ta W Re Os Ir Pt Au "
    "Hg Tl Pb Bi Po At Rn Fr Ra Ac Th Pa U Np Pu Am Cm Bk Cf Es Fm Md No Lr "
    "Rf Db Sg Bh Hs Mt Ds Rg Cn Nh Fl Mc Lv Ts Og".split()
)
_NAMES = dict(
    pair.split(":")
    for pair in (
        "Au:gold Ag:silver Pt:platinum Pd:palladium Ni:nickel Fe:iron Co:cobalt "
        "Cu:copper Zn:zinc Sn:tin Ti:titanium Si:silicon Al:aluminum Mn:manganese "
        "Cd:cadmium Pb:lead Mo:molybdenum W:tungsten Ru:ruthenium Rh:rhodium "
        "Ir:iridium Os:osmium Bi:bismuth Sb:antimony Se:selenium Te:tellurium"
    ).split()
)
_MATERIAL_WORDS = {
    "graphene",
    "graphite",
    "silica",
    "perovskite",
    "ceria",
    "titania",
    "alumina",
}
_GENERIC = set(
    "a an the this that these those i we you my your our to of for with in on at "
    "and or is are be can could should would how what why please make prepare "
    "synthesis synthesize synthesise synthesis protocol procedure attached "
    "attachment optimize optimise optimization optimisation target desired "
    "aspect ratio length diameter size nm mm ml material nanomaterial "
    "nanomaterials nanotechnology question help using use based change".split()
)
_FORMULA = re.compile(r"(?<!\w)(?:[A-Z][a-z]?\d*)+(?!\w)")
_MORPHOLOGY = r"(?:nano(?:wire|rod|particle|crystal|tube|sheet|cube)s?|alloy|oxide)s?"
_ALLOYS = {f"{a}{b}".lower(): f"{a}{b}" for a in _NAMES for b in _NAMES if a != b}
_SYMBOL_BY_NAME = {name: symbol for symbol, name in _NAMES.items()}


def _normalize(text: str) -> str:
    return unicodedata.normalize("NFKC", text or "")


def _formula_terms(text: str) -> list[str]:
    result = []
    for match in _FORMULA.finditer(_normalize(text)):
        formula = match.group()
        elements = re.findall(r"[A-Z][a-z]?", formula)
        if not all(element in _ELEMENTS for element in elements):
            continue
        if len(elements) == 1 and elements[0] not in _NAMES:
            continue
        if formula not in result:
            result.append(formula)
    return result


def material_scope(question: str, attachment_text: str = "") -> list[str]:
    """Prefer a named target over other reagents; infer only from this attachment."""
    question = _normalize(question)
    for text in (question, _normalize(attachment_text)):
        pairs = re.findall(
            rf"\b([A-Za-z]+)(?:\s+and\s+|[\s/–—-]+)([A-Za-z]+)\s+{_MORPHOLOGY}\b",
            text,
            flags=re.I,
        )
        for first, second in pairs:
            if first.lower() in _SYMBOL_BY_NAME and second.lower() in _SYMBOL_BY_NAME:
                return [
                    _SYMBOL_BY_NAME[first.lower()] + _SYMBOL_BY_NAME[second.lower()]
                ]
        targets = re.findall(
            rf"\b(\w+(?:[/-]\w+)*)\s+{_MORPHOLOGY}\b", text, flags=re.I
        )
        for target in targets:
            target = target.replace("/", "").replace("-", "")
            target = _ALLOYS.get(target.lower(), target)
            formulas = _formula_terms(target)
            if formulas:
                return formulas
        if text == question:
            formulas = _formula_terms(text)
            names = [
                name
                for name in _NAMES.values()
                if re.search(rf"\b{name}\b", text, re.I)
            ]
            words = [
                word
                for word in sorted(_MATERIAL_WORDS)
                if re.search(rf"\b{word}s?\b", text, re.I)
            ]
            if formulas or names or words:
                return formulas + names + words
    return []


def retrieval_question(question: str, scope: list[str]) -> str:
    missing = [
        term
        for term in scope
        if not re.search(rf"\b{re.escape(term)}\b", question, re.I)
    ]
    return " ".join([question, *missing]).strip()


def _matches_material(term: str, body: str) -> bool:
    if re.search(rf"(?<!\w){re.escape(term)}(?!\w)", body, re.I):
        return True
    elements = re.findall(r"[A-Z][a-z]?", term)
    if elements and "".join(elements) == term and all(e in _NAMES for e in elements):
        # PtNi, Pt-Ni, Pt/Ni, and platinum-nickel describe the same element pair.
        alternatives = [rf"(?:{element}|{_NAMES[element]})" for element in elements]
        return any(
            re.search(r"\b" + r"[\s/–—-]+".join(parts) + r"\b", body, re.I)
            for parts in (alternatives, list(reversed(alternatives)))
        )
    return False


def filter_relevant_hits(hits: list, question: str, scope: list[str]) -> list:
    """Filter before references and evidence are numbered; scores alone do not qualify."""
    terms = {
        word.lower() for word in re.findall(r"[A-Za-z][A-Za-z0-9]+", question)
    } - _GENERIC
    result = []
    for hit in hits:
        meta = (
            hit.get("meta", {}) if isinstance(hit, dict) else getattr(hit, "meta", {})
        )
        meta = meta if isinstance(meta, dict) else {}
        text = (
            (hit.get("text") or meta.get("text", ""))
            if isinstance(hit, dict)
            else getattr(hit, "text", "")
        )
        body = _normalize(
            " ".join(
                str(value or "")
                for value in (
                    meta.get("title"),
                    meta.get("abstract"),
                    text,
                )
            )
        )
        if not str(text or "").strip():
            continue
        if scope:
            keep = any(_matches_material(term, body) for term in scope)
        else:
            words = set(re.findall(r"[a-z][a-z0-9]+", body.lower()))
            keep = bool(terms & words)
        if keep:
            result.append(hit)
    return result
