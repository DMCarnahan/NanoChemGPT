
# NanoChemGPT "Decider" layer

This drops in a tiny judge that decides whether your KB already has enough evidence
to answer a question. If **yes**, you do normal RAG. If **no**, you enqueue the
text-miner (server-side only).

## Files

- `judge_sufficiency.py` – scoring + decision logic
- `intent.py` – micro intent classifier
- `kb.py` – thin wrappers + augmentation helpers (plug in your FAISS & store)
- `miner_queue.py` – stub queue enqueuer (replace with Celery/RQ/etc.)
- `app_patch_example.py` – a sample `/ask` route wired to the judge

## Integration (quick)
1. Copy the five files into your server package, e.g. `app/decider/`.
2. Replace the stubs in `kb.py` with your FAISS search and document store fetches.
3. Wire `ask_bp` into your Flask app (or copy the logic into your existing `/ask`).
4. Implement `enqueue_text_mining_job` in `miner_queue.py` to push jobs to your miner.
5. In your existing "RAG answer" path, pass the fetched JSON to your GPT caller.

## Tuning
- Start thresholds in `judge_sufficiency.THRESHOLD` at:
  - procedure: 0.60
  - definition: 0.45
  - comparison/mechanism: 0.55
- Adjust `SIM_FLOOR` (default 0.40) to your embedding scale.
- Log `reasons` for every request to calibrate with a small labeled set.

## Notes
- Do **not** expose any "miner" tool to the GPT. The judge runs server-side.
- Slot mapping in `extract_slots_present()` is permissive; tailor to your schema.
- `entity_hit_from_query_and_doc()` is a coarse sanity check; replace with your ontology resolver for best results.
