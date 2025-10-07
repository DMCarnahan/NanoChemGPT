# Reproducibility Playbook

This document describes the **single-command** path to reproduce NanoChemGPT’s reported results.

## System requirements

- OS: Linux (x86_64) or macOS; CI image may use `ubuntu-22.04`.
- Python: **3.11.x**
- GPU: *not required* for the released evals.
- Disk: ~5 GB for indices and cached models.

## One-command pipeline

```bash
make reproduce
```

This runs, in order:

1. `make setup` — creates a local venv and installs dependencies.
2. `make kb` — runs the harvester if `configs/harvest.yaml` exists.
3. `make index` — rebuilds retrieval indices (best-effort target; adjust as needed).
4. `make eval` — runs `ai_eval/grader.py` for span / span_attr / struct.
5. `make reports` — lists the generated `reports/report_*.json` artifacts.

## Determinism

- Recommendation: pin all library versions in `requirements.txt` and set seeds where applicable.
- Use `os.environ.get("PYTHONHASHSEED","0")` or export `PYTHONHASHSEED=0` in `.env` for stable hashing behavior if needed.

## Artifacts

Expected outputs (paths may vary):

- `runs/` — per-task logs.
- `reports/report_span.json`
- `reports/report_span_attr.json`
- `reports/report_struct.json`

## Troubleshooting

- **TypeError (pageSize int/str)** — sanitize YAML configs: cast numeric fields via `int(...)` in your loader.
- **Recursive JSON regex on 3.12** — use `import regex as re` and `(?R)` recursion instead of stdlib `re`.
- **401 on admin endpoints** — ensure `Authorization: Bearer $ADMIN_TOKEN` is set.

If a target fails because a config is missing, create the config (copy one of the provided examples), or skip that sub-eval.
