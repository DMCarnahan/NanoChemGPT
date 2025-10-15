from converter import convert_text_to_robot_ops
import json

SAMPLE = """1. **Procedure**:\n1. Dissolve 1.0 g NaCl in 10 mL water.\n2. Add 5 mL ethanol to the solution.\n3. Heat the mixture to 60 C for 30 minutes.\n4. Dry the product in an oven at 80 C for 2 h."""

doc = convert_text_to_robot_ops(SAMPLE)

print("=== MICRO_PLAN ===")
for i, op in enumerate(doc.get("micro_plan", [])):
    if op.get("verb") in ["wait", "set"]:
        print(f"{i}: {json.dumps(op, indent=2)}")

print("\n=== TIMING_DELAYS ===")
for d in doc.get("timing_delays", []):
    print(json.dumps(d, indent=2))

print("\n=== STEPS ===")
for i, step in enumerate(doc.get("steps", []), 1):
    print(f"\nStep {i}: {step.get('action')}")
    print(f"  ops: {[op.get('op') for op in step.get('ops', [])]}")
