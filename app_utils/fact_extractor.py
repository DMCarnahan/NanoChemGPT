"""Domain-specific fact extractor for method paragraphs."""
from __future__ import annotations

from typing import Dict
import re


def extract_facts_from_text(text: str) -> Dict:
    t = text.lower()

    def find(pat, flags=re.I):
        m = re.search(pat, text, flags)
        return (m.group(1) if m and m.groups() else (m.group(0) if m else "")) or ""

    conc_fe   = find(r"\b0\.1\s*m\b")
    conc_naoh = find(r"\b0\.45\s*m\b")
    vol_fe    = find(r"\b(\d+)\s*m[lL]\b\s*(?:iron|iron\(ii\)|iron solution)")
    temp_bath = find(r"(?:at|temperature(?:\s*was)?\s*at|temperature\s*=\s*)(\d+)\s*°?\s*c")
    rate_naoh = find(r"(\d+)\s*m[lL]\s*/?\s*min")
    term_ph   = find(r"pH\s*(\d+(?:\.\d+)?)")
    dry_temp  = find(r"dri(?:ed|ing)[^\.]*?(\d+)\s*°?\s*c")
    dry_time  = find(r"dri(?:ed|ing)[^\.]*?(\d+)\s*h")
    brand     = find(r"(mettler\s*toledo\s*dl50)", flags=re.I)

    water_bath  = "water bath" in t
    autotitr    = "autotitrator" in t or "auto-titrator" in t
    vac_filter  = "vacuum" in t and "filter" in t
    oven        = "oven" in t

    facts = {"hardware": [], "materials": [], "procedure": []}

    if water_bath: facts["hardware"].append(f"Water bath ({temp_bath or 'temperature not specified'})")
    if autotitr:   facts["hardware"].append(f"Autotitrator{f' ({brand})' if brand else ''}")
    if vac_filter: facts["hardware"].append("Vacuum filtration setup")
    if oven:       facts["hardware"].append("Oven")

    facts["materials"].append({
        "name": "iron(II) sulfate heptahydrate (FeSO4·7H2O)",
        "concentration": conc_fe or "not specified",
        "volume": (vol_fe + " mL") if vol_fe else "not specified",
        "role": "iron oxide precursor"
    })
    facts["materials"].append({
        "name": "sodium hydroxide (NaOH)",
        "concentration": conc_naoh or "not specified",
        "volume": "not specified",
        "role": "precipitating agent"
    })

    s1 = "Prepare the iron(II) sulfate solution"
    if conc_fe: s1 += f" ({conc_fe})"
    if vol_fe:  s1 += f", {vol_fe} mL"
    facts["procedure"].append(s1 + ".")

    step2 = "Maintain the reaction in a water bath"
    if temp_bath: step2 += f" at {temp_bath} °C"
    step2 += " with continuous stirring."
    facts["procedure"].append(step2)

    s3 = "Add NaOH"
    if conc_naoh: s3 += f" ({conc_naoh})"
    s3 += " to the iron solution"
    if rate_naoh: s3 += f" at {rate_naoh} mL/min"
    if autotitr:  s3 += " using an autotitrator"
    if brand:     s3 += f" ({brand})"
    s3 += ", monitoring pH."
    facts["procedure"].append(s3)

    if term_ph:
        facts["procedure"].append(f"Continue the titration to pH {term_ph}.")
    if vac_filter: facts["procedure"].append("Filter the precipitated solid under vacuum.")
    s6 = "Dry the solid in an oven"
    if dry_temp: s6 += f" at {dry_temp} °C"
    if dry_time: s6 += f" for {dry_time} h"
    s6 += "."
    facts["procedure"].append(s6)

    return facts
