import re

SAMPLE = """1. **Procedure**:\n1. Dissolve 0.5 g FeSO4·7H2O in 25 mL deionized water.\n2. Add 5 mL ethanol to the solution.\n3. Transfer the mixture to a clean beaker.\n4. Heat the mixture to 60 C for 30 minutes.\n5. Dry the product in an oven at 80 C for 2 h."""

print("=== RAW ===")
print(repr(SAMPLE))
print("\n=== LINES ===")
lines = SAMPLE.splitlines()
for i, line in enumerate(lines):
    print(f"{i}: '{line}'")

# Test step extraction logic
print("\n=== STEP EXTRACTION ===")
in_proc = False
steps = []
buf = []
for line in lines:
    if re.search(r"\*\*Procedure:?\*\*", line, re.I):
        print(f"Found procedure marker: '{line}'")
        in_proc = True
        continue
    if in_proc:
        if re.match(r"\s*\d+\.\s", line):
            if buf:
                steps.append(" ".join(buf).strip())
                print(f"  Completed step: {steps[-1][:60]}...")
                buf = []
            extracted = re.sub(r"^\s*\d+\.\s*", "", line).strip()
            print(f"  Starting new step from: '{line}' -> '{extracted}'")
            buf.append(extracted)
        else:
            if line.strip():
                print(f"  Continuing: '{line}'")
                buf.append(line.strip())

if buf:
    steps.append(" ".join(buf).strip())
    print(f"  Final step: {steps[-1][:60]}...")

print(f"\n=== FINAL STEPS ({len(steps)}) ===")
for i, step in enumerate(steps, 1):
    print(f"{i}. {step[:80]}")
