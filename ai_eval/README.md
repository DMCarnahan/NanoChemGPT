# NanoChem Eval Pro (mini harness)

This kit evaluates LLM outputs for chemistry information extraction with:
- **Span NER** (IoU or exact matching)
- **BIO token** scoring (plus entity-level from BIO)
- **Attribute rules** (co-occurrence windows, regex constraints, numeric ranges, typed units)
- **Relation extraction** (head–tail links, e.g., AMOUNT–UNIT, ACTION–MATERIAL)
- **Structured-output validation** (JSON schema-like checks, required keys/types)
- **Calibration** (optional Brier/ECE if predictions include probabilities)
- **Slices** (per-slice metrics by regex filters)
- **Error analysis** dumps

## Quick start

```bash
# Optional dependency (only used for full JSON Schema checks):
# pip install jsonschema PyYAML

cd ai_eval

# 1) NER span eval (IoU)
python3 grader.py -c configs/eval_span.yaml

# 2) Span eval with attribute rules
python3 grader.py -c configs/eval_span_attr.yaml

# 3) BIO token/ent eval
python3 grader.py -c configs/eval_bio.yaml

# 4) Relation extraction eval
python3 grader.py -c configs/eval_rel.yaml

# 5) Structured-output validation eval
python3 grader.py -c configs/eval_struct.yaml
```

## Datasets format (JSONL)

### Span NER
```json
{"id":"ex1","text":"...","spans":[{"start":10,"end":13,"label":"AMOUNT"}]}
```

### BIO
```json
{"id":"ex1","tokens":["In","a","100","mL","..."],"tags":["O","O","B-AMOUNT","B-UNIT","..."]}
```

### Relations
```json
{
  "id":"ex1",
  "text":"...",
  "entities":[{"start":10,"end":13,"label":"AMOUNT","eid":"e1"},{"start":14,"end":16,"label":"UNIT","eid":"e2"}],
  "relations":[{"head":"e1","tail":"e2","label":"MEASURED_IN"}]
}
```

### Structured output
```json
{
  "id":"ex1",
  "output": {
    "procedure":[{"action":"dissolve","material":"FeCl3·6H2O","amount":162,"unit":"mg"}],
    "hardware":["round-bottom flask","stir plate"]
  }
}
```

See `configs/*.yaml` for options.