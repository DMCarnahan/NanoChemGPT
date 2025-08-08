"""
step_validator.py — Parse, normalize, and validate lab step data from .txt or dict.

Features:
- Parses "Key: value" .txt files (case-insensitive keys).
- Remembers line numbers to report helpful validation errors.
- Normalizes units (ug→µg, ul→µL, umol→µmol, ml→mL, etc.).
- Normalizes vessel aliases (rbf/r b f→round-bottom flask, erlenmeyer→flask, etc.).
- Coerces numeric strings for amount/temperature/duration.
- Cleans reagent lists (trim, remove empties, dedupe preserving order).
- Validates against a strict JSON Schema (pairing rules, enums, ranges, regex).

Usage:
    from step_validator import validate_step, validate_file, ValidationError

    # From plain text:
    txt = "Action: Heat\nIdentity: Anneal Step\nReagents: NaCl, H2O\nVessel: rbf\n"
    data = validate_step(txt)  # returns normalized dict

    # From dict:
    data = validate_step({"action": "mix", "identity": "prep", "reagents": ["NaCl"], "vessel": "beaker"})

    # From file (auto-detect .json vs plain text):
    data = validate_file("step.txt")

    # Error handling:
    try:
        data = validate_step("Action: Mix\nVessel: Beaker 250 mL")
    except ValueError as e:
        print("Nice error:", e)

"""

import re
import json
from jsonschema import validate as _js_validate, ValidationError

# -------------------------
# 1) JSON Schema
# -------------------------
json_schema_regex = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "action": {"type": "string", "minLength": 1},
        "identity": {"type": "string", "minLength": 1},
        "reagents": {
            "type": "array",
            "items": {"type": "string", "minLength": 1},
            "minItems": 1,
            "uniqueItems": True
        },
        "solvent": {"type": "string", "minLength": 1},
        "amount": {"type": "number", "exclusiveMinimum": 0},
        "units": {
            "type": "string",
            "enum": [
                "g","mg","ug","µg","kg",
                "mL","L","uL","µL",
                "mol","mmol","umol","µmol"
            ]
        },
        "temperature": {"type": "number", "minimum": -273.15, "maximum": 1500},
        "duration": {"type": "number", "exclusiveMinimum": 0},
        "vessel": {
            "type": "string",
            "minLength": 1,
            "pattern": "^(beaker|flask|round-bottom flask|vial|dewar|schlenk flask)$"
        }
    },
    "required": ["action", "identity"],
    "allOf": [
        {"if": {"required": ["amount"]}, "then": {"required": ["units"]}},
        {"if": {"required": ["units"]}, "then": {"required": ["amount"]}},
        {"if": {"required": ["temperature"]}, "then": {"required": ["duration"]}},
        {"if": {"required": ["duration"]}, "then": {"required": ["temperature"]}}
    ]
}

# -------------------------
# 2) Normalizer
# -------------------------
def normalize_for_validation(data: dict) -> dict:
    """
    Normalize JSON data before schema validation.
    - Strips whitespace from all strings (recursively)
    - Lowercases 'vessel' and 'units'
    - Canonicalizes units (ug/μg/mcg→µg, ul/μl→µL, umol/μmol→µmol, ml→mL, etc.)
    - Numeric coercion for amount/temperature/duration if strings (supports sci. notation)
    - Normalizes vessel aliases (rbf / r b f → round-bottom flask, round bottom → round-bottom, schlenk → schlenk flask, erlenmeyer → flask)
    - Cleans reagent lists: trims, removes empties, deduplicates (preserving order)
    """
    unit_map = {
        "ug": "µg", "μg": "µg", "mcg": "µg", "microgram": "µg", "micrograms": "µg",
        "ul": "µL", "μl": "µL", "microliter": "µL", "microliters": "µL",
        "umol": "µmol", "μmol": "µmol", "micromole": "µmol", "micromoles": "µmol",
        "milliliter": "mL", "milliliters": "mL", "ml": "mL",
        "liter": "L", "liters": "L", "lt": "L",
        "gram": "g", "grams": "g",
        "milligram": "mg", "milligrams": "mg",
        "kilogram": "kg", "kilograms": "kg",
        "mole": "mol", "moles": "mol",
        "millimole": "mmol", "millimoles": "mmol",
        "molarity": "M", "molarities": "M",
        "millimolar": "mM", "millimolars": "mM",
        "percent": "%", "percentage": "%",
        "ppm": "ppm", "ppb": "ppb", "ppt": "ppt",
        "pH": "pH", "ph": "pH", "ph value": "pH"
    }
    vessel_map = {
        "rbf": "round-bottom flask",
        "round bottom flask": "round-bottom flask",
        "round‐bottom flask": "round-bottom flask",
        "round – bottom flask": "round-bottom flask",
        "erlenmeyer": "flask",
        "erlenmeyer flask": "flask",
        "schlenk": "schlenk flask",
        "schlenk tube": "schlenk flask",
        "rb flask": "round-bottom flask"
    }
    number_keys = {"amount", "temperature", "duration"}

    def coerce_number(val):
        if isinstance(val, (int, float)):
            return val
        if isinstance(val, str):
            s = val.strip().replace(",", "")
            if re.fullmatch(r"[+-]?(\d+(\.\d*)?|\.\d+)([eE][+-]?\d+)?", s):
                num = float(s)
                return int(num) if num.is_integer() else num
        return val

    def dedupe_preserve_order(seq):
        seen, out = set(), []
        for x in seq:
            if x not in seen:
                seen.add(x)
                out.append(x)
        return out

    def normalize_value(key, value):
        if isinstance(value, dict):
            return {k: normalize_value(k, v) for k, v in value.items()}
        if isinstance(value, list):
            norm_list = [normalize_value(key, v) for v in value]
            if key == "reagents":
                norm_list = [v for v in norm_list if isinstance(v, str) and v.strip() != ""]
                norm_list = dedupe_preserve_order(norm_list)
            return norm_list
        if isinstance(value, str):
            s = value.strip()
            if key in ("vessel", "units"):
                s = s.lower()
            if key == "units":
                s = unit_map.get(s, s)
            if key == "vessel":
                s = s.replace("–", "-").replace("—", "-").replace("‐", "-")
                s = re.sub(r"\s+", " ", s)
                compact = s.replace(" ", "")
                s = vessel_map.get(s, vessel_map.get(compact, s))
            if key in number_keys:
                return coerce_number(s)
            return s
        if key in number_keys:
            return coerce_number(value)
        return value

    return {k: normalize_value(k, v) for k, v in data.items()}

# -------------------------
# 3) Text Parser (line-aware)
# -------------------------
def parse_step_text_verbose(txt: str):
    """
    Parse a plain-text step with line numbers retained for better errors.
    Accepts lines like:
        Action: Heat
        Identity: Anneal Step
        Reagents: NaCl, H2O
        Solvent: water
        Amount: 250
        Units: ug
        Temperature: 80
        Duration: 30
        Vessel: R B F

    Unknown keys are ignored by default.
    """
    key_map = {
        "action": "action",
        "identity": "identity",
        "reagents": "reagents",
        "solvent": "solvent",
        "amount": "amount",
        "units": "units",
        "temperature": "temperature",
        "duration": "duration",
        "vessel": "vessel"
    }
    out, line_map = {}, {}
    for lineno, line in enumerate(txt.splitlines(), start=1):
        if ":" not in line:
            continue
        k, v = line.split(":", 1)
        k = k.strip().lower()
        v = v.strip()
        if not k:
            continue
        if k in key_map:
            ck = key_map[k]
            line_map[ck] = lineno
            if ck == "reagents":
                items = [s.strip() for s in re.split(r"[;,]", v)]
                out[ck] = items
            else:
                out[ck] = v
    return out, line_map

# -------------------------
# 4) Error pretty-printer
# -------------------------
def pretty_validation_error(err: ValidationError, line_map: dict) -> str:
    # Create a path like reagents[1] or vessel, etc.
    if err.path:
        parts = []
        for p in err.path:
            if isinstance(p, int):
                parts[-1] = f"{parts[-1]}[{p}]"
            else:
                parts.append(str(p))
        inst_path = ".".join(parts)
    else:
        inst_path = "(root)"

    # Attach line if we can
    line_info = ""
    top_key = err.path[0] if err.path else None
    if isinstance(top_key, int):
        top_key = None
    if top_key in line_map:
        line_info = f" (near line {line_map[top_key]})"

    # Special-case required field errors
    if err.validator == "required":
        m = re.search(r"'(.+?)' is a required property", err.message)
        if m:
            missing_key = m.group(1)
            hint_line = None
            for anchor in ("action", "identity", "reagents", "solvent", "vessel"):
                if anchor in line_map:
                    hint_line = line_map[anchor]
                    break
            if hint_line:
                return f"Missing required field '{missing_key}' around line {hint_line}."
            return f"Missing required field '{missing_key}'."

    return f"{err.message}{line_info} at {inst_path}."

# -------------------------
# 5) Public API
# -------------------------
def validate_step(obj) -> dict:
    """
    Accepts either:
      - a dict matching the schema structure, or
      - a text blob in the 'Key: value' format.

    Returns the normalized, validated dict.
    Raises ValueError with a human-friendly message on failure.
    """
    if isinstance(obj, str):
        data, line_map = parse_step_text_verbose(obj)
    elif isinstance(obj, dict):
        data, line_map = obj, {}
    else:
        raise TypeError("validate_step expects dict or str")

    norm = normalize_for_validation(data)
    try:
        _js_validate(instance=norm, schema=json_schema_regex)
    except ValidationError as e:
        raise ValueError(pretty_validation_error(e, line_map))
    return norm

def validate_file(path: str) -> dict:
    """
    Load a .txt (key: value lines) or .json file and validate.
    Returns normalized, validated dict.
    Raises ValueError with a human-friendly message on failure.
    """
    with open(path, "r", encoding="utf-8") as f:
        raw = f.read()

    # Try JSON first; otherwise treat as plain text
    try:
        loaded = json.loads(raw)
    except json.JSONDecodeError:
        loaded = raw  # plain text

    return validate_step(loaded)
