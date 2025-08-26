import subprocess, os, sys, json, time, logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)

def run_miner(payload: dict):
    """
    Background miner job. Keep it small & robust.
    """
    q = (payload or {}).get("query", "")
    logging.info("[miner] start for query=%r", q)

    root = Path(__file__).resolve().parents[1]  # project root (/app)
    harvester = root / "harvester" / "harvester.py"
    bundle = root / "harvester" / "out_auto" / "bundle.jsonl"
    merged  = root / "harvester" / "out_auto" / "bundle_with_methods.jsonl"
    add_fallback = root / "scripts" / "bundle_add_fallback.py"
    indexer = root / "retriever" / "index_jsonl.py"
    index_dir = root / "retriever" / "index"

    # 1) harvest
    cfg = f"out_dir: {str(root/'harvester'/'out_auto')}\nqueries:\n- {json.dumps(q)}\nsince_year: 2016\nmax_results_per_source: 6\n"
    cfg_path = root / f".miner_{int(time.time())}.yaml"
    cfg_path.write_text(cfg, encoding="utf-8")
    subprocess.check_call([sys.executable, str(harvester), "--config", str(cfg_path)])

    # 2) add fallback
    subprocess.check_call([sys.executable, str(add_fallback), str(bundle), str(merged)])

    # 3) index (use methods text)
    subprocess.check_call([
        sys.executable, str(indexer),
        "--bundle", str(bundle),
        "--index_dir", str(index_dir),
        "--text-key", "methods"
    ])

    logging.info("[miner] done for %r", q)
    return {"ok": True, "query": q}
