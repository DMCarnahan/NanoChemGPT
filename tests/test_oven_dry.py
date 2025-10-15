from converter import convert_text_to_robot_ops, detect_oven_dry
import json

text = """Procedure:\n1. Heat the mixture to 50 C.\n2. Dry the sample in an oven at 50 C for 1 hour.\n"""

print("=== DETECT OVEN DRY ===")
step2 = "Dry the sample in an oven at 50 C for 1 hour."
result = detect_oven_dry(step2)
print(f"Step: {step2}")
print(f"Result: {result}")

print("\n=== FULL CONVERSION ===")
doc = convert_text_to_robot_ops(text)

print("Steps:")
for i, step in enumerate(doc.get("steps", []), 1):
    print(f"  {i}. Action={step.get('action')}, Raw={step.get('raw', '')[:60]}")
    if step.get("ops"):
        print(f"     Ops: {[op.get('op') for op in step.get('ops', [])]}")

print("\nMicro plan (set ops only):")
for op in doc.get("micro_plan", []):
    if op.get("verb") == "set":
        print(f"  {json.dumps(op, indent=2)}")
