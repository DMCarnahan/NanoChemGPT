"""Turn model-generated protocol tables into explicit, lossless lists."""

from __future__ import annotations

import re


def _cells(line: str) -> list[str]:
    return re.split(r"(?<!\\)\|", line.strip().strip("|"))


def protocol_tables_to_lists(text: str) -> str:
    """Preserve each table's labels and values without selecting a condition.

    Only valid pipe tables with an alignment row are normalized. Fenced code and
    malformed rows remain verbatim so unsupported text is not silently lost.
    """
    lines = (text or "").splitlines()
    result = []
    index = 0
    fenced = False
    while index < len(lines):
        line = lines[index]
        if line.lstrip().startswith("```"):
            fenced = not fenced
        if not fenced and "|" in line and index + 1 < len(lines):
            headers = [cell.strip() for cell in _cells(line)]
            separators = [cell.strip() for cell in _cells(lines[index + 1])]
            if (
                len(headers) > 1
                and len(headers) == len(separators)
                and all(re.fullmatch(r":?-+:?", cell) for cell in separators)
            ):
                index += 2
                result.append("")
                while index < len(lines) and "|" in lines[index]:
                    values = [cell.strip() for cell in _cells(lines[index])]
                    if len(values) != len(headers):
                        break
                    result.append(
                        "- "
                        + "; ".join(
                            f"{header or f'Column {column}'}: {value}"
                            for column, (header, value) in enumerate(
                                zip(headers, values), start=1
                            )
                        )
                    )
                    index += 1
                result.append("")
                continue
        result.append(line)
        index += 1
    return "\n".join(result).strip()
