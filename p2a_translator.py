import requests, json, textwrap

ENDPOINT = "https://rxn4chemistry.ibmcloud.com/paragraph2actions/api/v1/translate"

def translate_paragraphs(paragraphs: list[str]) -> list[dict]:
    """Return P2A action objects for each paragraph (or {} on error)."""
    payload = {"paragraphs": paragraphs, "model": "p2a-v2"}  # v2 model, 2025
    r = requests.post(ENDPOINT, json=payload, timeout=30)
    r.raise_for_status()
    data = r.json()
    return data.get("actions", [{} for _ in paragraphs])
