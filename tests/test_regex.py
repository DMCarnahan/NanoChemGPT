import re

test1 = "Heat the mixture to 60 C for 30 minutes"

# Current pattern
pattern1 = r"(\d+(?:\.\d+)?)\s*(?:minute|min|mins|m)\b"
matches1 = list(re.finditer(pattern1, test1, re.I))
print(f"Current pattern matches: {[(m.group(0), m.group(1)) for m in matches1]}")

# Better pattern - only allow 'm' if followed by non-letter
pattern2 = r"(\d+(?:\.\d+)?)\s*(?:minute|min|mins)(?:\b|s\b)"
matches2 = list(re.finditer(pattern2, test1, re.I))
print(f"Better pattern matches: {[(m.group(0), m.group(1)) for m in matches2]}")

# Even better - just match the full words
pattern3 = r"(\d+(?:\.\d+)?)\s*(?:minutes?|mins?)\b"
matches3 = list(re.finditer(pattern3, test1, re.I))
print(f"Best pattern matches: {[(m.group(0), m.group(1)) for m in matches3]}")
