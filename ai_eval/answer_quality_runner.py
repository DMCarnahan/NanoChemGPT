"""Run a small, auditable live-model quality suite.

The extraction grader measures mining quality. This runner instead exercises
the exact answer prompt used by the web app against synthetic evidence cases.
It never reads production uploads or retrieval indexes.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app_utils.llm import (  # noqa: E402
    create_openai_client,
    generate_text,
    load_model_settings,
)
from app_utils.prompts import build_answer_prompt  # noqa: E402


def load_cases(path: Path) -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                case = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON on {path}:{line_number}: {exc}"
                ) from exc
            if not isinstance(case, dict) or not case.get("id"):
                raise ValueError(f"Case on {path}:{line_number} needs an id")
            cases.append(case)
    return cases


def grade_answer(answer: str, case: dict[str, Any]) -> dict[str, Any]:
    """Apply deterministic safety and format checks to one generated answer."""

    failures: list[str] = []
    for pattern in case.get("required_patterns") or []:
        if not re.search(pattern, answer, re.IGNORECASE | re.MULTILINE):
            failures.append(f"missing required pattern: {pattern}")
    for pattern in case.get("forbidden_patterns") or []:
        if re.search(pattern, answer, re.IGNORECASE | re.MULTILINE):
            failures.append(f"matched forbidden pattern: {pattern}")

    mode = case.get("mode", "protocol")
    if mode == "protocol" and "## Synthesis Protocol:" not in answer:
        failures.append("missing protocol heading")
    if mode == "reasoning" and "## Mechanistic reasoning" not in answer:
        failures.append("missing reasoning heading")
    if re.search(r"^##\s+References\b", answer, re.IGNORECASE | re.MULTILINE):
        failures.append("answer included a References section")

    allowed_citations = {
        int(value) for value in case.get("allowed_numeric_citations") or []
    }
    cited = {int(value) for value in re.findall(r"\[(\d+)\]", answer)}
    unexpected = cited - allowed_citations
    if unexpected:
        failures.append(f"invented numeric citations: {sorted(unexpected)}")

    return {
        "passed": not failures,
        "failures": failures,
        "cited": sorted(cited),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cases",
        type=Path,
        default=ROOT / "ai_eval" / "datasets" / "answer_quality_cases.jsonl",
    )
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--model", default=None)
    parser.add_argument("--reasoning-effort", default=None)
    args = parser.parse_args()

    client = create_openai_client()
    if client is None:
        parser.error("OPENAI_API_KEY is required for the live quality suite")

    settings = load_model_settings()
    model = args.model or settings.answer_model
    effort = args.reasoning_effort or settings.reasoning_effort
    results = []
    for case in load_cases(args.cases):
        prompt = build_answer_prompt(
            question=str(case["question"]),
            mode=str(case.get("mode") or "protocol"),
            context=str(case.get("context") or ""),
            references=str(case.get("references") or ""),
        )
        generated = generate_text(
            client,
            model=model,
            instructions=prompt.instructions,
            input_text=prompt.input_text,
            reasoning_effort=effort,
            max_output_tokens=settings.max_output_tokens,
        )
        grade = grade_answer(generated.text, case)
        results.append(
            {
                "id": case["id"],
                "model": generated.model,
                "response_id": generated.response_id,
                "answer": generated.text,
                **grade,
            }
        )

    passed = sum(1 for result in results if result["passed"])
    report = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "model": model,
        "reasoning_effort": effort,
        "passed": passed,
        "total": len(results),
        "pass_rate": passed / len(results) if results else 0.0,
        "results": results,
    }
    out = args.out or ROOT / "ai_eval" / "reports" / "answer_quality_latest.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(f"Answer quality: {passed}/{len(results)} passed -> {out}")
    return 0 if passed == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
