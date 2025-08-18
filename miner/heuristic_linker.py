"""
Heuristic relation linker: connect ACTION -> arguments (MATERIAL, TEMP, TIME, SPEED, AMOUNT/UNIT, CONC, VESSEL, ATMOS, EQUIPMENT)
Rules:
- Work sentence by sentence.
- For each ACTION span, attach the nearest argument spans within a window of +/- 1 sentence or 60 chars.
- Prefer arguments in the same sentence, then neighbors.
"""
from __future__ import annotations
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Any, Tuple
import spacy

@dataclass
class LinkedOp:
    action_text: str
    action_span: Tuple[int,int]
    sentence_id: int
    materials: List[str] = field(default_factory=list)
    temp: List[str] = field(default_factory=list)
    time: List[str] = field(default_factory=list)
    speed: List[str] = field(default_factory=list)
    conc: List[str] = field(default_factory=list)
    amounts: List[str] = field(default_factory=list)
    units: List[str] = field(default_factory=list)
    vessel: List[str] = field(default_factory=list)
    atmos: List[str] = field(default_factory=list)
    equipment: List[str] = field(default_factory=list)

LABEL_MAP = {
    "MATERIAL":"materials","TEMP":"temp","TIME":"time","SPEED":"speed","CONC":"conc",
    "AMOUNT":"amounts","UNIT":"units","VESSEL":"vessel","ATMOS":"atmos","EQUIPMENT":"equipment"
}

def link_doc(doc):
    sent_bounds = [(s.start_char, s.end_char) for s in doc.sents]
    def sent_id_for_span(a:int,b:int)->int:
        for i,(sa,sb) in enumerate(sent_bounds):
            if a>=sa and b<=sb: return i
        return min(range(len(sent_bounds)), key=lambda i: abs(a - sent_bounds[i][0])) if sent_bounds else 0

    ents = list(doc.ents)
    actions = [e for e in ents if e.label_=="ACTION"]
    others = [e for e in ents if e.label_!="ACTION"]

    by_sent: Dict[int, List] = {}
    for e in others:
        si = sent_id_for_span(e.start_char, e.end_char)
        by_sent.setdefault(si, []).append(e)

    linked = []
    for a in actions:
        si = sent_id_for_span(a.start_char, a.end_char)
        op = LinkedOp(action_text=a.text, action_span=(a.start_char,a.end_char), sentence_id=si)
        cand = by_sent.get(si, []) + by_sent.get(si-1, []) + by_sent.get(si+1, [])
        cand = sorted(cand, key=lambda e: abs(e.start_char - a.start_char))
        for e in cand:
            lab = e.label_
            field = LABEL_MAP.get(lab)
            if not field: 
                continue
            if abs(e.start_char - a.start_char) <= 60 or sent_id_for_span(e.start_char,e.end_char)==si:
                getattr(op, field).append(e.text)
        linked.append(asdict(op))
    return linked

if __name__ == "__main__":
    import sys, json
    nlp = spacy.blank("en"); nlp.add_pipe("sentencizer")
    text = sys.stdin.read()
    doc = nlp(text)
    doc.ents = []
    print(json.dumps(link_doc(doc), indent=2))