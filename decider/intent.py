import re
def classify_intent(q: str) -> str:
    s = q.lower()
    if re.search(r"\b(how (do|to)|procedure|synthesi[sz]e|protocol|recipe|step[- ]by[- ]step|make|prepare)\b", s):
        return "procedure"
    if re.search(r"\b(compare|better than|versus|vs\.?|trade[- ]offs?|state of the art|SOTA|benchmark)\b", s):
        return "comparison"
    if re.search(r"\b(why|mechanism|how does|what causes|origin of)\b", s):
        return "mechanism"
    if re.search(r"\b(what is|define|definition|explain)\b", s):
        return "definition"
    return "procedure"
