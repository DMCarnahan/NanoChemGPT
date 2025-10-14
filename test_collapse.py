from converter import convert_text_to_robot_ops
import json

FULL_TEXT = """**Procedure:**
5. Heat the mixture to 50 C for 4 minutes.
6. Continue heating at 50 C for 2 minutes.
7. Maintain temperature at 50 C while stirring vigorously.
"""

doc = convert_text_to_robot_ops(FULL_TEXT)

print("=== MICRO_PLAN (set ops only) ===")
for i, op in enumerate(doc.get("micro_plan", [])):
    if op.get("verb") == "set" and op.get("param") == "temperature_C":
        print(f"{i}: {json.dumps(op, indent=2)}")

print("\n=== MICRO_PLAN_MIN (50C set ops) ===")
min_sets = [a for a in doc.get('micro_plan_min', []) if a.get('verb')=='set' and a.get('param')=='temperature_C' and a.get('value')==50]
print(f"Count: {len(min_sets)}")
for op in min_sets:
    print(json.dumps(op, indent=2))
