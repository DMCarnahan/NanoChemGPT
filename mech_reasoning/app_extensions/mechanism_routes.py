from __future__ import annotations

import os
import regex as re
import json
import traceback
import _load_embed
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Tuple

from flask import Blueprint, request, jsonify, current_app

# RAG bits (mechanistic KB)
from retriever.retriever import search, Embedder

mechanism_bp = Blueprint("mechanism_bp", __name__, url_prefix="/mechanism")

class Embedder:
    def __init__(self, backend: str | None = None, model: str | None = None):
        store = _load_embed()
        self.backend = backend or (store and store.get("backend")) or "sentence-transformers"
        self.model = model or (store and store.get("model")) or "sentence-transformers/all-MiniLM-L6-v2"
        self._st = None  # lazy SentenceTransformer

    def encode(self, texts: list[str]) -> np.ndarray:
        if self.backend == "openai":
            from openai import OpenAI  # requires OPENAI_API_KEY
            client = OpenAI()
            resp = client.embeddings.create(model=self.model, input=texts)
            arr = np.array([d.embedding for d in resp.data], dtype="float32")
            arr /= (np.linalg.norm(arr, axis=1, keepdims=True) + 1e-8)
            return arr
        else:
            from sentence_transformers import SentenceTransformer
            if self._st is None:
                self._st = SentenceTransformer(self.model)
            vecs = self._st.encode(texts, normalize_embeddings=True)
            return vecs.astype("float32")

    # Some older code used .embed("text") instead of .encode([..])
    def embed(self, text: str) -> np.ndarray:
        return self.encode([text])[0]

# ---------- Prompt building ----------

def _root_dir() -> Path:
    return Path(__file__).resolve().parents[1]

def _template_path() -> Path:
    return _root_dir() / "prompts" / "mechanistic_answering_template.txt"

def build_prompt(question: str, hits: List[Dict[str, Any]]) -> str:
    """Render a lean, mechanism-focused prompt from the template + retrieved facts."""
    facts: List[str] = []
    # Pull compact parameter→effect lines; cap to keep prompt short.
    for h in hits:
        e = h.get("entry", {}) or {}
        system = e.get("system", "N/A")
        route = e.get("synthesis_method", "N/A")
        # Prefer effects that mention aspect_ratio/length/diameter
        prioritized, others = [], []
        for p in e.get("parameters", []) or []:
            pname = (p.get("name") or "").strip()
            role = (p.get("role") or "").strip()
            for eff in p.get("effects", []) or []:
                tgt = (eff.get("target") or "").strip().lower()
                line = f"{system}|{route}: {pname} -> {eff.get('target','')} ({eff.get('direction','')}); rationale: {eff.get('mechanistic_rationale','')}"
                if tgt in {"aspect_ratio", "length", "diameter"}:
                    prioritized.append(line)
                else:
                    others.append(line)
        facts.extend(prioritized + others)
    facts = facts[:12]

    try:
        tmpl = _template_path().read_text(encoding="utf-8")
    except Exception:
        # Minimal fallback template if file is missing
        tmpl = (
            f"System: N/A\nQuestion: {question}\nRoute: N/A\n\n"
            "Use ONLY the retrieved mechanistic facts below.\nFacts:\n"
            + "\n".join(f"- {f}" for f in facts)
            + "\n\nOutput JSON with keys: question, reasoning_steps, final_answer, scope, citations"
        )

    tmpl = tmpl.replace("{{ system }}", "N/A")
    tmpl = tmpl.replace("{{ question }}", question)
    tmpl = tmpl.replace("{{ synthesis_method }}", "N/A")
    facts_block = "\n".join(f"- {f}" for f in facts) if facts else "- (no facts retrieved)"
    tmpl = tmpl.replace("{% for f in facts %}\n  - {{ f }}\n  {% endfor %}", facts_block)

    # Hard-require strict JSON at the end
    strict_tail = (
        "\n\nIMPORTANT: Return ONLY a single JSON object, no prose, no markdown, no code fences.\n"
        '{"question":"'
        + question.replace('"', '\\"')
        + '","reasoning_steps":[],"final_answer":"","scope":"","citations":[]}\n'
        "Ensure valid JSON."
    )
    return tmpl + strict_tail


# ---------- Output parsing & validation ----------

_JSON_BLOCK_RX = re.compile(r"\{(?:[^{}]|(?R))*\}", re.M) 

ANSWER_SCHEMA = {
    "type": "object",
    "required": ["question", "reasoning_steps", "final_answer", "scope", "citations"],
    "properties": {
        "question": {"type": "string"},
        "reasoning_steps": {"type": "array", "items": {"type": "string"}},
        "final_answer": {"type": "string"},
        "scope": {"type": "string"},
        "citations": {"type": "array", "items": {"type": "string"}},
        "parameter_ranking": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["name", "score", "evidence_ids"],
                "properties": {
                    "name": {"type": "string"},
                    "score": {"type": "number"},
                    "evidence_ids": {"type": "array", "items": {"type": "string"}}
                }
            }
        }
    },
    "additionalProperties": True,
}

def _validate_answer_shape(obj: Dict[str, Any]) -> Tuple[bool, str]:
    # Lightweight manual validation 
    try:
        for k in ("question", "reasoning_steps", "final_answer", "scope", "citations"):
            if k not in obj:
                return False, f"missing key '{k}'"
        if not isinstance(obj["reasoning_steps"], list):
            return False, "'reasoning_steps' must be an array"
        if not isinstance(obj["citations"], list):
            return False, "'citations' must be an array"
        return True, ""
    except Exception as e:
        return False, f"validation error: {e}"

def _extract_json_object(raw: str) -> Dict[str, Any] | None:
    # 1) Try direct parse
    try:
        obj = json.loads(raw)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass
    # 2) Greedy extract the first JSON-like block
    m = _JSON_BLOCK_RX.search(raw)
    if m:
        try:
            obj = json.loads(m.group(0))
            if isinstance(obj, dict):
                return obj
        except Exception:
            pass
    return None

def _coerce_mechanistic_json(raw: str, question: str) -> Dict[str, Any]:
    obj = _extract_json_object(raw)
    if obj is not None:
        return obj
    # Fallback: fabricate minimal valid object from lines
    steps = [s.strip() for s in raw.splitlines() if s.strip()][:6]
    return {
        "question": question,
        "reasoning_steps": steps,
        "final_answer": raw.strip()[:1200],
        "scope": "unspecified",
        "citations": [],
    }


# ---------- Post-hoc helpers (citations & parameter ranking) ----------

def _collect_kb_citations(hits: List[Dict[str, Any]]) -> List[str]:
    """Extract DOI/URL-like citations from retrieved entries."""
    cites: List[str] = []
    seen = set()
    for h in hits:
        e = h.get("entry", {}) or {}
        for ev in e.get("evidence", []) or []:
            c = (ev.get("citation") or "").strip()
            u = (ev.get("url") or "").strip()
            key = c or u
            if key and key not in seen:
                cites.append(c or u)
                seen.add(key)
    return cites

def _score_parameters_from_hits(hits: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Rank parameters by their linkage to aspect_ratio/length/diameter across retrieved entries.
    Scoring heuristic:
      +2 if parameter has an effect with target 'aspect_ratio'
      +1 if target is 'length' or 'diameter'
      +0.5 bonus if direction is 'increase'/'decrease' (explicit)
      +0.5 if a mechanistic rationale string is present and non-trivial
    Results: [{name, score, evidence_ids: [entry ids]}]
    """
    scores: Dict[str, Dict[str, Any]] = {}
    for h in hits:
        e = h.get("entry", {}) or {}
        eid = e.get("id") or ""
        for p in e.get("parameters", []) or []:
            pname = (p.get("name") or "").strip()
            if not pname:
                continue
            base = scores.setdefault(pname, {"name": pname, "score": 0.0, "evidence_ids": set()})
            for eff in p.get("effects", []) or []:
                tgt = (eff.get("target") or "").strip().lower()
                direction = (eff.get("direction") or "").strip().lower()
                rationale = (eff.get("mechanistic_rationale") or "").strip()
                if tgt == "aspect_ratio":
                    base["score"] += 2.0
                elif tgt in {"length", "diameter"}:
                    base["score"] += 1.0
                if direction in {"increase", "decrease"}:
                    base["score"] += 0.5
                if len(rationale) >= 20:
                    base["score"] += 0.5
                if eid:
                    base["evidence_ids"].add(str(eid))
    ranked = sorted(scores.values(), key=lambda d: (-d["score"], d["name"].lower()))
    # Convert sets to lists for JSON
    for r in ranked:
        r["evidence_ids"] = sorted(list(r["evidence_ids"]))
    return ranked


# ---------- Route ----------

@mechanism_bp.route("/ask", methods=["POST"])
def ask():
    data = request.get_json(force=True)
    question = (data.get("question") or "").strip()
    k = int(data.get("k", 6) or 6)
    if not question:
        return jsonify({"error": "Missing 'question'"}), 400

    # Retrieve mechanistic facts
    try:
        hits = search(question, k=k, embedder=Embedder())
    except Exception as e:
        traceback.print_exc()
        hits = []

    prompt = build_prompt(question, hits)

    # 1) Call the model
    client = current_app.config.get("OPENAI_CLIENT")
    if client is None:
        return jsonify({"error": "OpenAI client not configured"}), 500
    try:
        raw = client.chat.completions.create(
            model=os.getenv("MECH_MODEL", "gpt-4o"),
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
        ).choices[0].message.content
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": f"LLM call failed: {e}"}), 500

    # 2) Coerce to JSON
    obj = _coerce_mechanistic_json(raw, question)

    # 3) Validate; if invalid, ask model to repair to strict JSON
    ok, why = _validate_answer_shape(obj)
    if not ok:
        try:
            repair_instr = (
                "Rewrite the previous answer STRICTLY as minified JSON with keys: "
                "question (string), reasoning_steps (array of strings), final_answer (string), "
                "scope (string), citations (array of strings). No prose, no markdown, JSON only."
            )
            repaired = client.chat.completions.create(
                model=os.getenv("MECH_MODEL", "gpt-4o"),
                messages=[
                    {"role": "user", "content": repair_instr},
                    {"role": "user", "content": raw},
                ],
                temperature=0,
            ).choices[0].message.content
            obj2 = _coerce_mechanistic_json(repaired, question)
            ok2, _ = _validate_answer_shape(obj2)
            if ok2:
                obj = obj2
        except Exception:
            pass

    # 4) Auto-append citations from KB if model returned none
    if isinstance(obj, dict) and isinstance(obj.get("citations"), list) and len(obj["citations"]) == 0:
        kb_cites = _collect_kb_citations(hits)
        if kb_cites:
            obj["citations"] = kb_cites[:10]

    # 5) Post-hoc parameter ranking (evidence-driven)
    ranking = _score_parameters_from_hits(hits)
    if ranking:
        obj["parameter_ranking"] = ranking

        # If final_answer doesn't clearly name a "most important" parameter, suggest one
        fa = (obj.get("final_answer") or "").lower()
        top = ranking[0]["name"] if ranking else None
        if top and top.lower() not in fa:
            suggestion = (
                f"Based on retrieved evidence, **{top}** is the leading candidate for the most "
                f"influential parameter (score={ranking[0]['score']:.2f})."
            )
            # Append to final_answer succinctly
            obj["final_answer"] = (obj.get("final_answer") or "")
            if obj["final_answer"] and not obj["final_answer"].endswith("\n"):
                obj["final_answer"] += "\n"
            obj["final_answer"] += suggestion

    # 6) Return everything (including raw model text for debugging)
    return jsonify({
        "question": question,
        "retrieved": [
            {
                "id": h.get("entry", {}).get("id"),
                "system": h.get("entry", {}).get("system"),
                "method": h.get("entry", {}).get("synthesis_method"),
            }
            for h in hits
        ],
        "prompt": prompt,
        "model_raw": raw,
        "answer": obj,
    })
