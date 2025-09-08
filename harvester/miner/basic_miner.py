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

# crude but effective chemical formula detector (needs at least one digit)
CHEMFORM_RX = re.compile(r"\b(?=.*\d)(?:[A-Z][a-z]?\d*){2,}\b")

# keywords that often flag materials even without a parser
MATERIAL_HINTS = {
    "oxide","chloride","sulfide","sulfate","nitrate","acetate",
    "hydroxide","carbonate","phosphate","boride","fluoride","iodide",
    "polymer","copolymer","nanoparticle","nanoparticles","nanotube",
    "nanowire","graphene","perovskite","mof","metal–organic","metal-organic",
    "alloy","ceramic","composite","solution","precursor","powder","film"
}

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
    def __init__(self, nlp_model: str | None = None):
        import os
        from pathlib import Path as _P
        # Resolve model path/name preference:
        cand = nlp_model or os.getenv('SPACY_MODEL')
        if cand == 'SPACY_MODEL':
            cand = os.getenv('SPACY_MODEL')
        if not cand:
            # Try local trained models first
            here = _P(__file__).parent
            local_ner = here / 'ner_model' / 'ner'
            local_best = here / 'ner_model' / 'model-best'
            if local_ner.exists():
                cand = str(local_ner)
            elif local_best.exists():
                cand = str(local_best)
            else:
                cand = 'en_core_web_sm'
        try:
            self.nlp = spacy.load(cand)
        except Exception as e:
            # Final fallback to small English model
            self.nlp = spacy.load('en_core_web_sm')

        self._lemma_ok = False
        self._ensure_pipeline_ready()

        if not any(p in self.nlp.pipe_names for p in ("parser", "senter", "sentencizer")):
            self.nlp.add_pipe("sentencizer")

        # Build action patterns; prefer LEMMA if lemmatizer is available
        self.matcher = Matcher(self.nlp.vocab)
        self._raw_patterns = []  # keep if you ever want to rebuild

        for label, verbs in ACTIONS.items():
            rule_id = label.upper()
            pats = []
            for v in verbs:
                if self._lemma_ok:
                    pats.append([{"LEMMA": v}])
                else:
                    pats.append([{"LOWER": v.lower()}])
            # register all patterns for this label at once
            self.matcher.add(rule_id, pats)
            self._raw_patterns.extend(pats)
        
        # precompile regexes
        self.rx_amount = re.compile(rf"\b({NUM_RX})\s*({UNIT_RX})\b", re.I)
        self.rx_temp = re.compile(TEMP_RX, re.I)
        self.rx_time = re.compile(TIME_RX, re.I)
        self.rx_speed = re.compile(SPEED_RX, re.I)
        self.rx_conc = re.compile(CONC_RX, re.I)

    def _ensure_pipeline_ready(self) -> None:
        """Guarantee doc.sents exists and try to enable LEMMA; set self._lemma_ok accordingly."""
        nlp = self.nlp

        # sentence boundaries
        if not any(p in nlp.pipe_names for p in ("parser", "senter", "sentencizer")):
            nlp.add_pipe("sentencizer")

        # Try to add a lemmatizer. Prefer lookup/rule if tables are available.
        try:
            if "attribute_ruler" not in nlp.pipe_names:
                nlp.add_pipe("attribute_ruler", first=True)
        except Exception:
            pass

        # Try rule mode first, then lookup mode
        added_lemma = False
        if "lemmatizer" not in nlp.pipe_names:
            try:
                nlp.add_pipe("lemmatizer", config={"mode": "rule"})
                added_lemma = True
            except Exception:
                try:
                    nlp.add_pipe("lemmatizer", config={"mode": "lookup"})
                    added_lemma = True
                except Exception:
                    pass

        try:
            nlp.initialize(lambda: [])
            self._lemma_ok = True
        except Exception:
            if added_lemma and "lemmatizer" in nlp.pipe_names:
                try:
                    nlp.remove_pipe("lemmatizer")
                except Exception:
                    pass
            self._lemma_ok = False

    def _lemma_free_patterns(self, patterns):
        """Convert patterns using LEMMA to LOWER so matching works without a lemmatizer."""
        def convert_token(tok):
            t = dict(tok)
            if "LEMMA" in t:
                # use case-insensitive surface form instead
                val = t.pop("LEMMA")
                if isinstance(val, str):
                    t["LOWER"] = val.lower()
                elif isinstance(val, list):
                    t["LOWER"] = [v.lower() for v in val]
            return t

        converted = []
        for pat in patterns:
            converted.append([convert_token(tok) for tok in pat])
        return converted

    def _finalize_matcher(self):
        """Rebuild matcher if lemma isn’t available, using LOWER instead of LEMMA."""
        if getattr(self, "_raw_patterns", None) is None:
            # nothing to do if you don't store the raw patterns
            return
        # rebuild the matcher based on whether lemmatization is working
        from spacy.matcher import Matcher
        self.matcher = Matcher(self.nlp.vocab)
        if self._lemma_ok:
            self.matcher.add("OPERATION", self._raw_patterns)
        else:
            self.matcher.add("OPERATION", self._lemma_free_patterns(self._raw_patterns))

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

    def _guess_materials(self, sent_doc):
        """
        Try to extract material-like mentions from a sentence doc.
        Works even if the pipeline lacks a dependency parser.
        """
        mats = set()

        # a) take entities that look like materials (if your NER has such labels)
        for ent in getattr(sent_doc, "ents", ()):
            lbl = (ent.label_ or "").lower()
            if lbl in {"material", "chemical", "compound", "reagent"}:
                mats.add(ent.text)

        # b) use noun_chunks if (and only if) a parser is available
        if "parser" in self.nlp.pipe_names:
            try:
                for nc in sent_doc.noun_chunks:
                    # heuristics: keep chunks with material-ish keywords
                    if any(tok.lower_ in MATERIAL_HINTS for tok in nc):
                        mats.add(nc.text)
            except Exception:
                pass
        else:
            # c) no parser → fallback: scan tokens for keyword windows
            toks = list(sent_doc)
            i = 0
            while i < len(toks):
                t = toks[i]
                if (t.is_alpha and t.lower_ in MATERIAL_HINTS):
                    # grow a small window around the keyword
                    start = max(0, i - 2)
                    end = min(len(toks), i + 3)
                    span = sent_doc[start:end]
                    mats.add(span.text)
                    i = end
                    continue
                i += 1

        # d) formulas like TiO2, H2SO4, FeCl3, etc.
        for m in CHEMFORM_RX.finditer(sent_doc.text):
            mats.add(m.group(0))

        # keep it tidy
        out = sorted(mats, key=lambda s: (len(s), s.lower()))[:6]
        return out
    # cap to keep outputs tidy

    def extract(self, text: str) -> list[dict]:
        doc = self.nlp(text)
        results = []
        for sent in doc.sents:
            # run pipeline on sentence text to ensure all attrs (including lemma) are set
            subdoc = self.nlp(sent.text)
            for match_id, start, end in self.matcher(subdoc):
                span = subdoc[start:end]
                label = self.nlp.vocab.strings[match_id].lower()
                results.append({
                    "op_type": label,
                    "sentence": sent.text,
                    "start_char": span.start_char + sent.start_char,
                    "end_char": span.end_char + sent.start_char,
                    "materials": self._guess_materials(subdoc),
                    "params": self._extract_params(sent.text),
                })
        return results

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

    def extract_procedure(self, text: str) -> dict:
        ops = self.extract(text)
        expanded = self.expand(ops)
        return {'operations': ops, 'expanded': expanded}
