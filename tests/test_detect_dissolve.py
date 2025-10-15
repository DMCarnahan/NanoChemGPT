from converter import detect_dissolve

test_steps = [
    "Dissolve 0.5 g FeSO4·7H2O in 25 mL deionized water.",
    "Add 5 mL ethanol to the solution.",
    "Transfer the mixture to a clean beaker.",
    "Heat the mixture to 60 C for 30 minutes.",
    "Dry the product in an oven at 80 C for 2 h."
]

for i, step in enumerate(test_steps, 1):
    result = detect_dissolve(step)
    print(f"Step {i}: {step[:50]}")
    print(f"  Dissolve detected: {result}")
    print()
