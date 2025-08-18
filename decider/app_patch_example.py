
"""
Example /ask route showing how to wire the decider.
Replace the kb_search/kb_fetch implementations and integrate with your GPT answerer.
"""
from __future__ import annotations
from flask import Blueprint, request, jsonify
from typing import List, Dict
from .intent import classify_intent
from .kb import kb_search, kb_fetch, extract_slots_present, infer_year, extract_source_domain, entity_hit_from_query_and_doc
from .judge_sufficiency import judge_sufficiency
from .miner_queue import enqueue_text_mining_job

ask_bp = Blueprint("ask", __name__)

def _augment_hits(query: str, raw_hits: List[Dict]) -> List[Dict]:
    """
    Take raw hits from kb_search and enrich them for the judge.
    We fetch each doc's normalized JSON to compute slots_present and entity_hit.
    """
    ids = [h["id"] for h in raw_hits]
    docs = {d["id"]: d for d in kb_fetch(ids)}
    aug = []
    for h in raw_hits:
        d = docs.get(h["id"], {})
        meta = d.get("meta", {})
        aug.append({
            "id": h["id"],
            "sim": float(h.get("sim", 0.0)),
            "year": infer_year(meta),
            "source_domain": extract_source_domain(meta.get("source") or meta.get("url") or meta.get("host")),
            "slots_present": list(extract_slots_present(d)),
            "entity_hit": entity_hit_from_query_and_doc(query, d),
        })
    return aug

@ask_bp.route("/ask", methods=["POST"])
def ask():
    payload = request.get_json(force=True)
    q = payload.get("query","").strip()
    user_id = payload.get("user_id")

    if not q:
        return jsonify({"error":"missing query"}), 400

    intent = classify_intent(q)
    raw_hits = kb_search(q, topk=8)

    aug_hits = _augment_hits(q, raw_hits)
    score, decision, reasons = judge_sufficiency(aug_hits, intent=intent)

    if decision == "use_kb":
        # Pull top context and answer via RAG (your existing path)
        top_ids = [h["id"] for h in raw_hits[:5]]
        context = kb_fetch(top_ids)
        # TODO: call your GPT answerer with "context"
        return jsonify({
            "mode": "kb",
            "intent": intent,
            "sufficiency": reasons,
            "context_count": len(context),
            "answer": "[TODO] call answer_with_rag()",
        })

    # Otherwise, not enough evidence — enqueue mining
    job_id = enqueue_text_mining_job(q, user_id=user_id, intent=intent, reason=reasons.get("reason"), features=reasons)
    # Option A: return a concise partial answer from whatever we have (if you implement it)
    # Option B: return an acknowledgement now; UI can show a 'Mining…' badge
    return jsonify({
        "mode": "mining_enqueued",
        "intent": intent,
        "sufficiency": reasons,
        "job_id": job_id,
        "message": "Not enough high-quality evidence in KB; enqueued text-mining.",
    }), 202
