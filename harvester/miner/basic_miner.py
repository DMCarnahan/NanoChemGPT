from __future__ import annotations
from dataclasses import dataclass, field, asdict
import re
import spacy
from spacy.matcher import Matcher
from typing import List, Dict, Any, Optional

# --- Patterns ---
UNIT_RX = r"(?:mg|g|kg|µg|ug|mL|L|µL|ul|mol|mmol|µmol|M|wt%|vol%)"
NUM_RX = r"\d+(?:\.\d+)?"
TEMP_RX = r"(-?\d+(?:\.\d+)?)\s*°\s?[CKF]|(-?\d+(?:\.\d+)?)\s*K\b"
TIME_RX = r"(\d+(?:\.\d+)?)\s*(?:s|sec|secs|second|seconds|min|mins|minute|minutes|h|hr|hrs|hour|hours)\b"
SPEED_RX = r"(\d+(?:\.\d+)?)\s*(?:rpm)\b"
CONC_RX = r"(\d+(?:\.\d+)?)(?:\s*(?:M|mM|µM|%|wt%|vol%))\b"

ACTIONS = {
    "add": ["add", "added", "adding", "introduce", "charge", "charged"],
    "stir": ["stir", "stirred", "stirring", "agitate", "mix", "mixed"],
    "heat": ["heat", "heated", "heating", "maintain", "maintained", "anneal", "annealed", "reflux"],
    "degas": ["degas", "degassed", "purge", "purged", "sparge", "sparged"],
    "inject": ["inject", "injected", "injection"],
    "wash": ["wash", "washed", "rinse", "rinsed"],
    "centrifuge": ["centrifuge", "centrifuged", "centrifugation"],
    "dry": ["dry", "dried", "drying", "evaporate", "evaporated"],
    "sonicate": ["sonicate", "sonicated", "sonication"],
    "filter": ["filter", "filtered", "filtration"],
}

@dataclass
class Operation:
    op_type: str
    sentence: str
    start_char: int
    end_char: int
    materials: List[str] = field(default_factory=list)
    params: Dict[str, Any] = field(default_factory=dict)

class BasicMiner:
    def __init__(self, spacy_model: str = "en_core_web_sm"):
        self.nlp = spacy.load(spacy_model, disable=["ner"])
        self.matcher = Matcher(self.nlp.vocab)
        # action matcher: single verbs via lexicon
        for label, verbs in ACTIONS.items():
            for v in verbs:
                self.matcher.add(label.upper(), [[{"LEMMA": v}]])
        # precompile regexes
        self.rx_amount = re.compile(rf"\b({NUM_RX})\s*({UNIT_RX})\b", re.I)
        self.rx_temp = re.compile(TEMP_RX, re.I)
        self.rx_time = re.compile(TIME_RX, re.I)
        self.rx_speed = re.compile(SPEED_RX, re.I)
        self.rx_conc = re.compile(CONC_RX, re.I)

    def _extract_params(self, text: str) -> Dict[str, Any]:
        p: Dict[str, Any] = {}
        if m := self.rx_temp.search(text):
            p["temp"] = m.group(0).replace(" ", "")
        if m := self.rx_time.search(text):
            p["time"] = m.group(0)
        if m := self.rx_speed.search(text):
            p["speed"] = m.group(0)
        # amounts/concs can appear multiple times
        p["amounts"] = [m.group(0) for m in self.rx_amount.finditer(text)]
        p["concs"] = [m.group(0) for m in self.rx_conc.finditer(text)]
        return {k: v for k, v in p.items() if v}

    def _guess_materials(self, sent_doc) -> List[str]:
        mats = []
        for chunk in sent_doc.noun_chunks:
            # heuristic: chemical-looking tokens (formula fragments, parentheses, numerals)
            if re.search(r"[A-Za-z]+\d|\(|\)|[··]|[IVX]+", chunk.text) or any(t.like_num for t in chunk):
                mats.append(chunk.text.strip())
        # de-dup, short filter
        mats = [m for m in dict.fromkeys(mats) if len(m) > 2]
        return mats[:4]  # cap to keep outputs tidy

    def extract(self, text: str) -> List[Dict[str, Any]]:
        doc = self.nlp(text)
        results: List[Operation] = []
        for sent in doc.sents:
            matches = self.matcher(sent.as_doc())
            if not matches:
                continue
            # choose the first action label (simplest)
            label = self.nlp.vocab.strings[matches[0][0]]
            op_type = label.lower()
            params = self._extract_params(sent.text)
            materials = self._guess_materials(sent)
            results.append(Operation(
                op_type=op_type,
                sentence=sent.text,
                start_char=sent.start_char,
                end_char=sent.end_char,
                materials=materials,
                params=params
            ))
        return [asdict(r) for r in results]

    # --- Expansion to low-level micro-steps ---
    def expand(self, ops: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        expanded = []
        for op in ops:
            t = op["op_type"]
            mats = op.get("materials") or []
            p = op.get("params") or {}
            vessel = "reaction vessel"
            if t == "add":
                amt = (p.get("amounts") or ["the specified amount"])[0]
                mat = mats[0] if mats else "reagent"
                steps = [
                    f"Ensure {vessel} is present and empty.",
                    "Place a clean stir bar in the vessel.",
                    f"Measure {amt} of {mat}.",
                    f"Transfer {mat} into the vessel."
                ]
            elif t == "stir":
                speed = p.get("speed", "an appropriate speed")
                time_ = p.get("time", "the specified time")
                steps = [
                    f"Place the {vessel} on a stir plate.",
                    f"Set the stirrer to {speed}.",
                    f"Stir for {time_}."
                ]
            elif t == "heat":
                temp = p.get("temp", "the target temperature")
                time_ = p.get("time", "the specified time")
                steps = [
                    f"Place the {vessel} on a hotplate.",
                    f"Ramp to {temp}.",
                    f"Maintain {temp} for {time_}."
                ]
            else:
                steps = [f"Perform operation '{t}' as described: {op['sentence']}"]
            expanded.append({"op_type": t, "steps": steps})
        return expanded
