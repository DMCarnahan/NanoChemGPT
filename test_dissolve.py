from converter import convert_text_to_robot_ops
import json

# Test for dissolve
SAMPLE = """1. **Procedure**:\n1. Dissolve 0.5 g FeSO4·7H2O in 25 mL deionized water.\n2. Add 5 mL ethanol to the solution.\n3. Transfer the mixture to a clean beaker.\n4. Heat the mixture to 60 C for 30 minutes.\n5. Dry the product in an oven at 80 C for 2 h."""

doc = convert_text_to_robot_ops(SAMPLE)

print("=== STEPS ===")
for i, step in enumerate(doc.get("steps", []), 1):
    print(f"\nStep {i}:")
    print(f"  Action: {step.get('action')}")
    print(f"  Raw: {step.get('raw', '')[:80]}")
    if step.get("reagents_structured"):
        print(f"  Reagents: {[r.get('name') for r in step.get('reagents_structured', [])]}")

# Test for oven drying
text2 = """Procedure:\n1. Heat the mixture to 50 C.\n2. Dry the sample in an oven at 50 C for 1 hour.\n"""
doc2 = convert_text_to_robot_ops(text2)

print("\n\n=== DRYING TEST ===")
print("Steps:")
for i, step in enumerate(doc2.get("steps", []), 1):
    print(f"  Step {i}: {step.get('action')}")
    print(f"  Ops: {[op.get('op') for op in step.get('ops', [])]}")

print("\nMicro plan:")
for op in doc2.get("micro_plan", []):
    if op.get("verb") == "set" and op.get("device") in ["OV1", "oven"]:
        print(f"  {json.dumps(op, indent=4)}")
