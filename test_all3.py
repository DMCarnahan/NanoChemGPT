from converter import convert_text_to_robot_ops
import json

FULL_TEXT = """**Procedure:**
5. Heat the mixture to 50 C for 4 minutes.
6. Continue heating at 50 C for 2 minutes.
7. Maintain temperature at 50 C while stirring vigorously.
"""

doc = convert_text_to_robot_ops(FULL_TEXT)

print("=== STEPS ===")
for i, step in enumerate(doc.get("steps", []), 1):
    print(f"{i}. Action={step.get('action')}")
    if step.get("ops"):
        for op in step["ops"]:
            print(f"   - {op.get('op')}: {op}")

print("\n=== MICRO_PLAN (all ops) ===")
for i, op in enumerate(doc.get("micro_plan", [])):
    print(f"{i}: verb={op.get('verb')}, device={op.get('device')}, param={op.get('param')}, value={op.get('value')}, step_index={op.get('step_index')}")
