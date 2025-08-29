SHELL := /bin/bash
PY ?= python3
ENVFILE ?= .env

# Export env from .env if present
ifneq (,$(wildcard $(ENVFILE)))
include $(ENVFILE)
export $(shell sed -n 's/^\([A-Za-z_][A-Za-z0-9_]*\)=.*/\1/p' $(ENVFILE))
endif

# ---- Paths you can tweak ----
HARVEST_CONFIG ?= configs/harvest.yaml
INDEX_DIR      ?= data/index
MECH_INDEX_DIR ?= data/mech_index

SPAN_CFG       ?= configs/eval_span.json
SPAN_ATTR_CFG  ?= configs/eval_span_attr.json
STRUCT_CFG     ?= configs/eval_struct.json

APP_MODULE     ?= app:app
HOST           ?= 0.0.0.0
PORT           ?= 8000

.PHONY: help setup venv kb index index-text index-mech eval eval-span eval-span-attr eval-struct demo health reproduce reports clean admin-rebuild

help:
	@echo "Targets:"
	@echo "  setup           — create venv and install requirements.txt"
	@echo "  kb              — run harvester/harvester.py with $(HARVEST_CONFIG)"
	@echo "  index           — build all indices (text + mechanistic)"
	@echo "  index-text      — build TF-IDF/FAISS text index via retriever/index_jsonl.py"
	@echo "  index-mech      — build mechanistic FAISS (best-effort; skips if module missing)"
	@echo "  eval            — run all evals (span, span_attr, struct)"
	@echo "  demo            — launch gunicorn on $(HOST):$(PORT)"
	@echo "  admin-rebuild   — call /admin/rebuild_mech_index with ADMIN_TOKEN"
	@echo "  reproduce       — setup -> kb -> index -> eval -> reports"
	@echo "  reports         — list reports/report_*.json"
	@echo "  clean           — remove caches and build artifacts"

venv:
	@$(PY) -m venv .venv
	@. .venv/bin/activate; pip install -U pip

setup: venv
	@. .venv/bin/activate; pip install -r requirements.txt

kb:
	@if [ -f $(HARVEST_CONFIG) ]; then \
		echo "[kb] Running harvester with $(HARVEST_CONFIG)"; \
		. .venv/bin/activate; $(PY) harvester/harvester.py --config $(HARVEST_CONFIG); \
	else \
		echo "[kb] Missing $(HARVEST_CONFIG) — create it or override HARVEST_CONFIG=..."; exit 1; \
	fi

index: index-text index-mech

index-text:
	@echo "[index-text] Building text index into $(INDEX_DIR)"
	@if [ -f retriever/index_jsonl.py ]; then \
		. .venv/bin/activate; $(PY) retriever/index_jsonl.py --index_dir $(INDEX_DIR); \
	else \
		echo "[index-text] retriever/index_jsonl.py not found — skipping"; \
	fi

index-mech:
	@echo "[index-mech] Building mechanistic index (best-effort)"
	@if [ -f mech_reasoning/retriever/retriever.py ]; then \
		. .venv/bin/activate; $(PY) - <<'PY'\
import os, sys, importlib
try:
    m = importlib.import_module("mech_reasoning.retriever.retriever")
    if hasattr(m, "build_mechanistic_index"):
        print("[index-mech] Calling build_mechanistic_index() ...")
        m.build_mechanistic_index(index_dir=os.environ.get("MECH_INDEX_DIR","data/mech_index"))
    else:
        print("[index-mech] No build_mechanistic_index() in module; try running module as a script")
        os.system(f"{sys.executable} mech_reasoning/retriever/retriever.py || true")
except Exception as e:
    print("[index-mech] Skipped due to error:", e)
PY\
	; else \
		echo "[index-mech] mech_reasoning/retriever/retriever.py not found — skipping"; \
	fi

eval: eval-span eval-span-attr eval-struct

eval-span:
	@if [ -f ai_eval/grader.py ] && [ -f $(SPAN_CFG) ]; then \
		. .venv/bin/activate; $(PY) ai_eval/grader.py -c $(SPAN_CFG); \
	else \
		echo "[eval-span] Missing ai_eval/grader.py or $(SPAN_CFG)"; exit 1; \
	fi

eval-span-attr:
	@if [ -f ai_eval/grader.py ] && [ -f $(SPAN_ATTR_CFG) ]; then \
		. .venv/bin/activate; $(PY) ai_eval/grader.py -c $(SPAN_ATTR_CFG); \
	else \
		echo "[eval-span-attr] Missing $(SPAN_ATTR_CFG) — skipping"; \
	fi

eval-struct:
	@if [ -f ai_eval/grader.py ] && [ -f $(STRUCT_CFG) ]; then \
		. .venv/bin/activate; $(PY) ai_eval/grader.py -c $(STRUCT_CFG); \
	else \
		echo "[eval-struct] Missing $(STRUCT_CFG) — skipping"; \
	fi

reports:
	@echo "---- Latest reports (if any) ----"
	@find reports -maxdepth 1 -type f -name 'report_*.json' -print 2>/dev/null || true
	@echo "---------------------------------"

demo:
	@. .venv/bin/activate; gunicorn -w 2 -k gthread --threads 2 --timeout 90 \
		--worker-tmp-dir /dev/shm -b $(HOST):$(PORT) $(APP_MODULE)

health:
	@curl -fsS http://$(HOST):$(PORT)/healthz && echo " OK" || (echo " FAIL" && exit 1)

admin-rebuild:
	@if [ -z "$$ADMIN_TOKEN" ]; then echo "Set ADMIN_TOKEN in .env"; exit 1; fi
	@curl -fsS -X POST http://$(HOST):$(PORT)/admin/rebuild_mech_index \
		-H "Authorization: Bearer $$ADMIN_TOKEN" \
		-H "Content-Type: application/json" \
		-d '{}' && echo ""

reproduce: setup kb index eval reports

clean:
	@rm -rf .venv __pycache__ .pytest_cache build dist *.egg-info \
		runs/* reports/* $(INDEX_DIR)/* $(MECH_INDEX_DIR)/* 2>/dev/null || true
