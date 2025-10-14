from converter import find_minutes, _clean_unicode
import re

test1 = "Heat the mixture to 60 C for 30 minutes"
clean = _clean_unicode(test1)
print(f"Original: '{test1}'")
print(f"Clean: '{clean}'")
print(f"Minutes found: {find_minutes(test1)}")

# Test the regex directly
pattern = r"(\d+(?:\.\d+)?)\s*(?:minute|min|mins|m)\b"
matches = list(re.finditer(pattern, clean, re.I))
print(f"Regex matches: {[(m.group(0), m.group(1)) for m in matches]}")
