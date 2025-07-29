from __future__ import annotations
import json, re, math, textwrap
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Callable, Tuple

__all__ = ["convert_to_json", "ParserError"]
SCHEMA_VERSION = "1.8.2"

class ParserError(ValueError):
    pass

_HEADING_LINE = re.compile(
    r"""^\s*
        (?:\d+[\.\)]\s*)?
        (?:\*\*)?
        (?P<name>hardware(?:\s*&\s*glassware)?|glassware|materials|reagents|procedure|steps|method)
        (?:\*\*)?
        \s*:?\s*$
    """, re.I | re.VERBOSE)

_LIST_BULLET = re.compile(r"^\s*(?:[-*•–—]\s+|\d+[\.\)]\s+)")
_AMOUNT_RX = re.compile(r"(?P<qty>\d+(?:\.\d+)?)\s*(?P<unit>mg|g|kg|µl|μl|ul|mL|ml|L|l)\b", re.I)
_CONC_RX   = re.compile(r"(?P<val>\d+(?:\.\d+)?)\s*(?P<unit>mM|M|%(?:\s*w\/v|\s*v\/v)?)\b", re.I)
_VOL_RX    = re.compile(r"(?P<val>\d+(?:\.\d+)?)\s*(?P<vunit>mL|L)\b", re.I)

_REASONING_MARKERS = re.compile(r"\b(because|since|so that|therefore|thus|hence|rationale|justif(?:y|ication)|ensuring)\b", re.I)
_META_RX = re.compile(r"^\s*(this (protocol|procedure)|these steps|the following (procedure|protocol)|which can be|overview|background|this will serve|this step is crucial|ensure stability)\b", re.I)
_STRIP_CIT_RX = re.compile(r"\s*\[(?:CTX|GEN|\d+)\]\s*$", re.I)

_TRAIL_REASON_RX = re.compile(r"\s*(?:[,;\.\-–—]\s*)?(?:to\s+(?:ensure|facilitate|promote|allow|prevent|remove|minimi[sz]e|improve|enhance)\b.*|ensuring\b.*)$", re.I)
_SECOND_SENTENCE_REASON_RX = re.compile(r"\.\s*(?:This|These|It is|Note that)\b[^.]{0,200}?(?:crucial|important|helps?|ensur(?:e|es)|to\s+(?:prevent|remove|minimi[sz]e|enhance)).*$", re.I)

_VESSEL_PATTERNS: List[Tuple[re.Pattern, str]] = [
    (re.compile(r"schlenk flask", re.I), "schlenk_flask"),
    (re.compile(r"round[- ]?bottom flask|rbf", re.I), "flask"),
    (re.compile(r"\bflask\b", re.I), "flask"),
    (re.compile(r"\bbeaker\b", re.I), "beaker"),
    (re.compile(r"\bvial\b", re.I), "vial"),
    (re.compile(r"centrifuge tube|eppendorf|falcon", re.I), "tube"),
    (re.compile(r"\btube\b", re.I), "tube"),
    (re.compile(r"\breaction vessel|reactor|autoclave|teflon-lined", re.I), "reactor"),
]
_SIZE_RX = re.compile(r"(\d+(?:\.\d+)?)\s*(mL|L)\b", re.I)
_SEPARATE_VESSEL_RX = re.compile(r"\b(?:in(?:to)?\s+(?:a|an)\s+)(?:(?P<vol>\d+(?:\.\d+)?)\s*(?P<vu>mL|L)\s+)?(?P<typ>schlenk flask|round[- ]?bottom flask|flask|beaker|vial|centrifuge tube|tube)\b", re.I)

_PREPARE_SOLUTION_RX = re.compile(r"\b(prepare|make|formulate)\s+(?P<amt>\d+(?:\.\d+)?)\s*(?P<aunit>mg|g|kg)\s+(?P<chem>[^,;\.\n]+?)\s+in\s+(?P<vol>\d+(?:\.\d+)?)\s*(?P<vunit>mL|L)\s+(?P<solv>[^.;\n]+)", re.I)
_PH_RX = re.compile(r"pH\s*(?:of\s*)?(?:around\s*)?(?P<val>\d+(?:\.\d+)?)", re.I)

_STIR_SPEED_RXS = [
    re.compile(r"\bstir(?:ring)?\s*(?:at|speed\s*(?:to|=)?)\s*(?P<rpm>\d{2,4})\s*rpm\b", re.I),
    re.compile(r"\bset\s*stirr(?:er|ing)\s*(?:speed\s*(?:to|=)?)?\s*(?P<rpm>\d{2,4})\s*rpm\b", re.I),
]

_SETUP_DEDUP_RXS = [
    re.compile(r"^insert stir bar into vessel$", re.I),
    re.compile(r"^place vessel on magnetic stir plate$", re.I),
    re.compile(r"^turn on stir plate to target speed$", re.I),
]

_RCF_VALUE_RX = re.compile(r"(?P<rcf>\d{3,6})\s*(?:×\s*)?(?:g|xg)\b|\brcf\b\s*(?P<rcf2>\d{3,6})", re.I)
_RADIUS_RX = re.compile(r"(?:radius|r\s*=?)\s*(?P<rad>\d+(?:\.\d+)?)\s*cm", re.I)
_CENT_RPM_GLOBAL = re.compile(r"centrifug\w*[^.]*?\bat\s*(?P<rpm>\d{3,5})\s*rpm", re.I)
_CENT_TIME_GLOBAL = re.compile(r"centrifug\w*[^.]*?\bfor\s*(?P<val>\d+(?:\.\d+)?)\s*(?P<u>min|mins|minutes|minute|h|hr|hrs|hour|hours|s|sec|secs|seconds)\b", re.I)

def _rcf_to_rpm(rcf: float, r_cm: float) -> int:
    return int(round((rcf / (1.118e-5 * r_cm)) ** 0.5))

def _extract_rcf_and_radius(text: str):
    rcf = None; rad = None
    m = _RCF_VALUE_RX.search(text)
    if m:
        s = m.group("rcf") or m.group("rcf2")
        try: rcf = float(s)
        except: rcf = None
    rm = _RADIUS_RX.search(text)
    if rm:
        try: rad = float(rm.group("rad"))
        except: rad = None
    return rcf, rad

def _extract_centrifuge_time(text: str) -> Optional[str]:
    m = _CENT_TIME_GLOBAL.search(text)
    if m:
        val, u = m.group("val"), m.group("u").lower()
        if u.startswith("h"): return f"{val} h"
        if u.startswith("s"): return f"{val} s"
        return f"{val} min"
    return None

def _extract_centrifuge_params(text: str):
    m = _CENT_RPM_GLOBAL.search(text)
    rpm = int(m.group("rpm")) if m else None
    t = _extract_centrifuge_time(text)
    if rpm: return rpm, t
    rcf, rad = _extract_rcf_and_radius(text)
    if rcf is not None:
        rpm = _rcf_to_rpm(rcf, rad if rad is not None else 11.0)
    return rpm, t

def _protocol_window(text: str) -> str:
    m = re.search(r"^\s*##\s*SynthesisProtocol\b.*", text, flags=re.I | re.M | re.S)
    return text[m.start():] if m else text

def _canon(name: str) -> str:
    n = name.lower()
    if "hardware" in n or "glassware" in n: return "hardware"
    if n in ("materials", "reagents"): return "reagents"
    return "procedure"

def _split_sections_tolerant(text: str) -> Dict[str, List[str]]:
    text = _protocol_window(text)
    buckets: Dict[str, List[str]] = {"hardware": [], "reagents": [], "procedure": []}
    current: Optional[str] = None
    for raw in text.splitlines():
        line = raw.rstrip()
        if not line.strip(): continue
        norm = re.sub(r"[\*_`]", "", line).strip()
        m = _HEADING_LINE.match(norm)
        if m: current = _canon(m.group("name")); continue
        if current: buckets[current].append(raw)
        else: buckets.setdefault("_pre", []).append(raw)
    return buckets

def _bullets(lines: List[str]) -> List[str]:
    if not lines: return []
    any_bullets = any(_LIST_BULLET.match(l) for l in lines)
    items: List[str] = []
    for l in lines:
        s = l.strip()
        if not s: continue
        if any_bullets: s = _LIST_BULLET.sub("", s).strip()
        items.append(s)
    return items

@dataclass
class Reagent:
    description: str
    amount: Optional[float] = None
    unit: Optional[str] = None
    concentration: Optional[str] = None
    final_volume_mL: Optional[float] = None
    def asdict(self): return asdict(self)

def _parse_reagent(line: str) -> Reagent:
    amt  = _AMOUNT_RX.search(line)
    conc = _CONC_RX.search(line)
    vol  = _VOL_RX.search(line)
    return Reagent(
        description=line,
        amount=float(amt.group("qty")) if amt else None,
        unit=(amt.group("unit") if amt else None),
        concentration=(conc.group(0) if conc else None),
        final_volume_mL=(float(vol.group("val")) if vol else None),
    )

Action = List[str]
RuleFn = Callable[[re.Match], Action]

def _centrifuge_actions_from_match(m: re.Match) -> List[str]:
    rpm_val: Optional[int] = None
    if m.groupdict().get("rpm"):
        try: rpm_val = int(m.group("rpm"))
        except: rpm_val = None
    if rpm_val is None:
        rcf, rad = _extract_rcf_and_radius(m.group(0))
        if rcf is not None:
            rpm_val = _rcf_to_rpm(rcf, rad if rad is not None else 11.0)
    tstr: Optional[str] = None
    if m.groupdict().get("tval"):
        u = (m.group("tunit") or "").lower()
        tval = m.group("tval")
        tstr = f"{tval} h" if u.startswith("h") else f"{tval} s" if u.startswith("s") else f"{tval} min"
    rpm_txt = f"set speed {rpm_val} rpm" if rpm_val else "set speed 900 rpm"
    time_txt = f"run for {tstr}" if tstr else "run for 10 min"
    return ["load tubes into centrifuge", rpm_txt, time_txt, "pour off supernatant into waste, keep pellet"]

ACTION_RULES: List[Tuple[re.Pattern, RuleFn]] = [
    (re.compile(r"\bstir(?:red|ring)?\b", re.I),
     lambda m: ["insert stir bar into vessel", "place vessel on magnetic stir plate", "turn on stir plate to target speed"]),
    (re.compile(r"\badd(?:ed)?\s+(.*)", re.I),
     lambda m: [f"add {m.group(1).strip()} to vessel"]),
    (re.compile(r"\btransfer(?:red)?\s+(.*)\s+to\s+(.*)", re.I),
     lambda m: [f"transfer {m.group(1).strip()} to {m.group(2).strip()}"]),
    (re.compile(r"\bheat(?:ed)?\s+(?:the\s+)?(?:mixture|solution|suspension)\s*to\s*(?P<T>\d+(?:\.\d+)?)\s*°?\s*C(?:[^\.]*?)\bfor\s*(?P<tval>\d+(?:\.\d+)?)\s*(?P<tunit>min|mins|minutes|h|hr|hrs|hour|hours)\b", re.I),
     lambda m: [f"set heating device to {m.group('T')} °C", "monitor temperature until set point reached", f"hold temperature at {m.group('T')} °C for {m.group('tval')} {m.group('tunit')}"]),
    (re.compile(r"\bheat(?:ed)?\s+to\s+(?P<T>\d+(?:\.\d+)?)\s*°?\s*C\b", re.I),
     lambda m: [f"set heating device to {m.group('T')} °C", "monitor temperature until set point reached"]),
    (re.compile(r"\b(?:cool|allow|let)\b(?:ed)?\s+(?:the\s+)?(?:mixture|solution|reaction|vessel)?\s*(?:to\s*)?(?:room\s*temperature|r\.?t\.?)\b", re.I),
     lambda m: ["cool vessel to 25 °C"]),
    (re.compile(r"\bcool(?:ed)?\s+to\s+(?P<T>\d+(?:\.\d+)?)\s*°?\s*C\b", re.I),
     lambda m: [f"cool vessel to {m.group('T')} °C"]),
    (re.compile(r"\bfilter(?:ed|ation)?\b", re.I),
     lambda m: ["assemble filtration apparatus", "pass mixture through filter", "collect solid on filter"]),
    (re.compile(r"\bcentrifug(e|ed|ation)\b.*?(?:(?:at|@)\s*(?P<rpm>\d{2,5})\s*rpm)?(?:.*?(?:rcf\b|\d{3,6}\s*(?:×\s*)?g|\d{3,6}\s*xg).*?)?(?:.*?\bfor\s*(?P<tval>\d+(?:\.\d+)?)(?:\s*(?P<tunit>min|mins|minutes|minute|h|hr|hrs|hour|hours|s|sec|secs|seconds))?)?", re.I),
     _centrifuge_actions_from_match),
    (re.compile(r"\b(purge|degass?|bubble)\s+(with\s+)?(n2|nitrogen|argon|ar)\b", re.I),
     lambda m: ["connect inert gas line to vessel", "open gas flow to purge headspace", "maintain flow for specified duration"]),
    (re.compile(r"\b(dry|evaporate)\b.*\b(vacuum|vac)\b", re.I),
     lambda m: ["place sample in vacuum chamber", "apply vacuum until solvent removed or mass constant"]),
    (re.compile(r"\badjust\s+pH\s+to\s+(\d+(?:\.\d+)?)", re.I),
     lambda m: ["measure solution pH", f"add acid/base to reach pH {m.group(1)}", "verify pH is stable"]),
]

def _expand_procedure_line(line: str) -> List[str]:
    for rx, fn in ACTION_RULES:
        m = rx.search(line)
        if m: return fn(m)
    return [line]

def _strip_reasoning_fragments(s: str) -> str:
    s = _STRIP_CIT_RX.sub("", s)
    s = re.sub(r"^\s*finally,\s*", "", s, flags=re.I)
    s = re.sub(r"\((?:[^)]{0,80})\)", "", s)  # drop short parenthetical
    s = _TRAIL_REASON_RX.sub("", s)
    s = re.sub(r"[:–—-]\s*(?=(because|since|so that|therefore|thus|hence)\b).*", "", s, flags=re.I)
    s = re.sub(r"\b(because|since|so that|therefore|thus|hence|rationale|justif(?:y|ication)|ensuring).*$", "", s, flags=re.I)
    s = re.sub(r"\.\s*to vessel\s*$", "", s, flags=re.I)
    return s.strip()

def _detect_vessel_type(s: str) -> Optional[str]:
    for rx, vtype in _VESSEL_PATTERNS:
        if rx.search(s): return vtype
    return None

def _collect_vessels_from_hardware(hardware: List[str]) -> List[Dict[str, str]]:
    vessels: List[Dict[str, str]] = []
    for h in hardware:
        vtype = _detect_vessel_type(h or "")
        if vtype:
            size = _SIZE_RX.search(h or "")
            desc = (f"{size.group(0)} " if size else "") + vtype.replace("_", " ")
            vid = f"V{len(vessels)+1}"
            vessels.append({"id": vid, "type": vtype, "description": desc.strip()})
    if not vessels:
        vessels.append({"id": "V1", "type": "flask", "description": "flask"})
    return vessels

def _assign_targets(steps: List[str], vessels: List[Dict[str, str]]) -> List[Dict[str, str]]:
    current = vessels[0]["id"]
    out: List[Dict[str, str]] = []
    for s in steps:
        m = _SEPARATE_VESSEL_RX.search(s)
        if m:
            vol = (m.group("vol") or "").strip()
            vu  = (m.group("vu")  or "").strip()
            typ = (m.group("typ") or "").strip().lower()
            desc = (f"{vol} {vu} " if vol and vu else "") + typ
            vid = f"V{len(vessels)+1}"
            vessels.append({"id": vid, "type": typ, "description": desc})
            current = vid
        if re.search(r"\bcentrifug", s, re.I):
            tube = next((v for v in vessels if "tube" in v["description"].lower()), None)
            if not tube:
                vid = f"V{len(vessels)+1}"; vessels.append({"id": vid, "type": "tube", "description": "tube"})
                tube = vessels[-1]
            current = tube["id"]
        out.append({"action": s, "target": current})
    return out

def _dedupe_adjacent(seq: List[str]) -> List[str]:
    out = []
    last = None
    for s in seq:
        if s != last:
            out.append(s); last = s
    return out

def _global_dedupe_setup(seq: List[str]) -> List[str]:
    seen = [False, False, False]
    out: List[str] = []
    for s in seq:
        low = s.strip().lower()
        matched = False
        for i, rx in enumerate(_SETUP_DEDUP_RXS):
            if rx.match(low):
                matched = True
                if not seen[i]:
                    out.append(s); seen[i] = True
                break
        if not matched:
            out.append(s)
    return out

def _extract_stir_speed(text: str) -> int:
    for rx in _STIR_SPEED_RXS:
        m = rx.search(text)
        if m:
            try:
                rpm = int(m.group("rpm"))
                if 20 <= rpm <= 5000:
                    return rpm
            except ValueError:
                pass
    return 900

def convert_to_json(raw: str, robot: bool = False) -> Dict[str, object]:
    if not raw or not raw.strip():
        raise ParserError("Input text is empty.")
    raw = textwrap.dedent(raw).strip()

    # Split sections
    def _canon(name: str) -> str:
        n = name.lower()
        if "hardware" in n or "glassware" in n: return "hardware"
        if n in ("materials", "reagents"): return "reagents"
        return "procedure"

    # Reuse tolerant splitter from above
    def _split_sections_tolerant(text: str) -> Dict[str, List[str]]:
        m = re.search(r"^\s*##\s*SynthesisProtocol\b.*", text, flags=re.I | re.M | re.S)
        if m: text = text[m.start():]
        buckets: Dict[str, List[str]] = {"hardware": [], "reagents": [], "procedure": []}
        current: Optional[str] = None
        for rawl in text.splitlines():
            line = rawl.rstrip()
            if not line.strip(): continue
            norm = re.sub(r"[\*_`]", "", line).strip()
            mh = _HEADING_LINE.match(norm)
            if mh: current = _canon(mh.group("name")); continue
            if current: buckets[current].append(rawl)
            else: buckets.setdefault("_pre", []).append(rawl)
        return buckets

    def _bullets(lines: List[str]) -> List[str]:
        if not lines: return []
        any_bullets = any(_LIST_BULLET.match(l) for l in lines)
        items: List[str] = []
        for l in lines:
            s = l.strip()
            if not s: continue
            if any_bullets: s = _LIST_BULLET.sub("", s).strip()
            items.append(s)
        return items

    sections = _split_sections_tolerant(raw)
    hardware_lines  = _bullets(sections.get("hardware", []))
    reagent_lines   = _bullets(sections.get("reagents", []))
    procedure_lines = _bullets(sections.get("procedure", []))
    if not any([hardware_lines, reagent_lines, procedure_lines]):
        procedure_lines = _bullets(sections.get("_pre", [])) or _bullets(raw.splitlines())

    reagents = []
    for l in reagent_lines:
        amt  = _AMOUNT_RX.search(l)
        conc = _CONC_RX.search(l)
        vol  = _VOL_RX.search(l)
        reagents.append({
            "description": l,
            "amount": float(amt.group("qty")) if amt else None,
            "unit": (amt.group("unit") if amt else None),
            "concentration": (conc.group(0) if conc else None),
            "final_volume_mL": (float(vol.group("val")) if vol else None),
        })

    expanded: List[str] = []
    for line in procedure_lines:
        m = re.search(r"\badjust\s+pH\s+to\s+(\d+(?:\.\d+)?)", line, re.I)
        if m:
            steps = ["measure solution pH", f"adjust with acid/base to reach pH {m.group(1)}", "verify pH is stable"]
            expanded.extend(steps); continue
        for rx, fn in ACTION_RULES:
            mm = rx.search(line)
            if mm: expanded.extend(fn(mm)); break
        else:
            expanded.append(line)

    # Vessels
    vessels: List[Dict[str, str]] = _collect_vessels_from_hardware(hardware_lines)

    final_actions = expanded
    if robot:
        tmp: List[str] = []
        for s in expanded:
            if _META_RX.search(s):  # drop meta/prose
                continue
            s2 = _strip_reasoning_fragments(s).rstrip(".")
            if s2: tmp.append(s2)
        tmp = _dedupe_adjacent(tmp)
        tmp = _global_dedupe_setup(tmp)
        final_actions = tmp

        chosen_rpm = _extract_stir_speed(raw) or _extract_stir_speed(" ".join(procedure_lines))
        final_actions = [re.sub(r"\bturn on stir plate to target speed\b", f"set stirrer to {chosen_rpm} rpm", a, flags=re.I) for a in final_actions]

        rpm, ctime = _extract_centrifuge_params(raw)
        if rpm or ctime:
            new = []
            for a in final_actions:
                a2 = re.sub(r"\bset speed (?:as specified|900 rpm)\b", f"set speed {rpm or 900} rpm", a, flags=re.I)
                if ctime:
                    a2 = re.sub(r"\brun for (?:specified time|10 min)\b", f"run for {ctime}", a2, flags=re.I)
                new.append(a2)
            final_actions = new

    structured = _assign_targets(final_actions if robot else expanded, vessels)
    out: Dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "title"         : raw.split("\n", 1)[0].lstrip("# ")[:120],
        "hardware"      : hardware_lines,
        "reagents"      : reagents,
        "procedure"     : final_actions if robot else expanded,
    }
    if robot:
        out["vessels"] = vessels
        out["procedure_structured"] = structured
        out["robot"] = {"cleaned": True, "vessel_labels": True, "rpm_default": 900}
    if not out["procedure"]:
        raise ParserError("No steps recognized.")
    return out

if __name__ == "__main__":
    import sys
    txt = open(sys.argv[1], "r", encoding="utf-8").read()
    print(json.dumps(convert_to_json(txt, robot=True), indent=2, ensure_ascii=False))