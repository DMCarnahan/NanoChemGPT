import json
import re
from pathlib import Path


def ensure_dir(p):
    Path(p).mkdir(parents=True, exist_ok=True)


def write_json(p, obj):
    p = Path(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


def safe_slug(s, maxlen=80):
    return re.sub(r"[^a-zA-Z0-9._-]+", "_", s)[:maxlen]
