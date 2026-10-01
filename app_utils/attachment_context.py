"""Bound attachment evidence without dropping complete short protocols."""

from __future__ import annotations

from .request_validation import truncate_context
from .text_chunks import best_chunks_from_text


def _allocate_context(sizes: list[int], budget: int) -> list[int]:
    """Keep small documents whole, sharing the remaining space among larger ones."""
    allocations = [0] * len(sizes)
    pending = list(range(len(sizes)))
    while pending:
        share = budget // len(pending)
        complete = [i for i in pending if sizes[i] <= share]
        if not complete:
            for i in pending:
                allocations[i] = share
            break
        for i in complete:
            allocations[i] = sizes[i]
            budget -= sizes[i]
            pending.remove(i)
    return allocations


def build_attachment_context(
    attachments: list[tuple[str, str]], question: str, *, maximum_chars: int
) -> tuple[str, list[dict]]:
    """Include full text when it fits; mark larger, source-ordered excerpts."""
    if not attachments:
        return "", []
    headers = [
        f"[A{j}.1] attachment:{aid}\n"
        for j, (aid, _text) in enumerate(attachments, start=1)
    ]
    texts = [text.strip() for _aid, text in attachments]
    available = max(0, maximum_chars - sum(map(len, headers)) - 2 * (len(texts) - 1))
    allocations = _allocate_context([len(text) for text in texts], available)
    blocks, metadata = [], []
    marker = "[Attachment excerpted by server; other document text is omitted.]\n"
    for (aid, _text), header, text, capacity in zip(
        attachments, headers, texts, allocations
    ):
        excerpted = len(text) > capacity
        if excerpted:
            remaining = max(0, capacity - len(marker))
            chunks = best_chunks_from_text(
                text,
                question,
                max_chunk_chars=min(1200, max(1, remaining)),
                top_k=max(1, remaining // 1202),
                preserve_order=True,
            )
            content = marker + truncate_context(
                "\n\n".join(chunks) or text, maximum_chars=remaining
            )
            content = content[:capacity]
        else:
            content = text
        blocks.append(header + content)
        metadata.append(
            {
                "id": aid,
                "source_chars": len(text),
                "context_chars": len(content),
                "excerpted": excerpted,
            }
        )
    return "\n\n".join(blocks), metadata
