from converter import convert_text_to_robot_ops
import json

FULL_TEXT = """**Procedure:**
7. Maintain temperature at 50 C while stirring vigorously.
"""

doc = convert_text_to_robot_ops(FULL_TEXT)

print("=== STEPS ===")
for i, step in enumerate(doc.get("steps", []), 1):
    print(f"{i}. Action={step.get('action')}, Raw={step.get('raw', '')[:60]}")
    print(f"   Ops: {step.get('ops', [])}")

print("\n=== MICRO_PLAN ===")
for i, op in enumerate(doc.get("micro_plan", [])):
    print(f"{i}: {json.dumps(op, indent=2)}")
