from converter import find_minutes

test1 = "Heat the mixture to 60 C for 30 minutes"
test2 = "Dry for 2 h"

print(f"Test 1: '{test1}' -> {find_minutes(test1)} min")
print(f"Test 2: '{test2}' -> {find_minutes(test2)} min")
