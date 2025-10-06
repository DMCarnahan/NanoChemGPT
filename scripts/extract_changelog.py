#!/usr/bin/env python3
"""Extract the first subsection under [Unreleased] from CHANGELOG.md.

Usage: python scripts/extract_changelog.py --path CHANGELOG.md
Prints the extracted content to stdout.
"""

from pathlib import Path
import argparse
import re


def extract(text: str) -> str:
    m = re.search(r"^##\s*\[Unreleased\](.*?)(?=^##\s*\[)", text, flags=re.S | re.M)
    if not m:
        return ""
    block = m.group(1)
    # find first '###' subsection
    sub = re.search(r"(?m)^###\s*.*$", block)
    if sub:
        start = sub.start()
        rest = block[start + 1 :]
        next_sub = re.search(r"(?m)^###\s*.*$", rest)
        if next_sub:
            end = start + 1 + next_sub.start()
            content = block[start:end].strip()
        else:
            content = block[start:].strip()
    else:
        p = re.search(r"(.*?)(?:\n\n|$)", block, flags=re.S)
        content = p.group(1).strip() if p else block.strip()
    return content


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--path", default="CHANGELOG.md")
    args = p.parse_args()
    path = Path(args.path)
    if not path.exists():
        print("")
        return
    txt = path.read_text(encoding="utf-8")
    out = extract(txt)
    print(out)


if __name__ == "__main__":
    main()
