"""Rendering utilities to convert extracted facts into markdown."""

from __future__ import annotations

from typing import Dict, List


def bullets(xs: List[str]) -> str:
    return "\n".join([f"- {x}" for x in xs])


def render_protocol_md(facts: Dict) -> str:
    parts = []
    if facts.get("materials"):
        parts.append("## Materials")
        for m in facts["materials"]:
            parts.append(
                f"- **{m.get('name')}** — {m.get('role')} ({m.get('concentration')}, {m.get('volume')})"
            )
    if facts.get("hardware"):
        parts.append("\n## Equipment")
        parts.append(bullets(facts.get("hardware", [])))
    if facts.get("procedure"):
        parts.append("\n## Procedure")
        parts.append(bullets(facts.get("procedure", [])))
    return "\n\n".join(parts)
