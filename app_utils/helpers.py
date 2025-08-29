"""Minimal helper implementations to stop 500s and enable measurement."""
from __future__ import annotations
import html, json, re
from dataclasses import dataclass, asdict
from glob import glob
from pathlib import Path
from typing import Iterable, Optional, List

def classify_intent(q: str) -> str:
    if not q: return "reason"
    ql = q.lower()
    robot_kw = ["make","synthesize","synthesis","protocol","procedure","recipe","step-by-step","robot mode","robot-mode","structured json"]
    reason_kw = ["why","how","mechanism","explain","rationale","compare","tradeoff"]
    admin_kw = ["rebuild","index","harvest","admin","healthz","status"]
    convert_kw = ["convert","to json","export","download","schema"]
    eval_kw = ["eval","evaluate","score","precision","recall","f1"]
    viz_kw = ["figure","graphic","image","diagram","plot"]
    def any_in(t, keys): return any(k in t for k in keys)
    if any_in(ql, robot_kw):  return "robot"
    if any_in(ql, admin_kw):  return "admin"
    if any_in(ql, convert_kw):return "convert"
    if any_in(ql, eval_kw):   return "eval"
    if any_in(ql, viz_kw):    return "viz"
    if any_in(ql, reason_kw): return "reason"
    return "reason"

STOPWORDS = {"the","a","an","and","or","of","in","on","for","to","with","by","at","as","is","are","was","were","be","been","it","that","this","these","those","from","we","our","their","into","over","under","about","after","before","between"}

@dataclass
class Hit:
    text: str
    score: float
    meta: dict
    def asdict(self): return asdict(self)

def _tokenize(s: str) -> List[str]:
    return [w for w in re.findall(r"[A-Za-z0-9_]+", s.lower()) if w not in STOPWORDS]

def _yield_jsonl_chunks(path: Path):
    try:
        with path.open("r", encoding="utf-8", errors="ignore") as f:
            for i, line in enumerate(f, start=1):
                try:
                    obj = json.loads(line)
                    if isinstance(obj, dict):
                        fields = [obj[k] for k in ("title","abstract","content","text","body","raw") if isinstance(obj.get(k), str)]
                        txt = "\n".join(fields) if fields else json.dumps(obj, ensure_ascii=False)
                    else:
                        txt = json.dumps(obj, ensure_ascii=False)
                except Exception:
                    txt = line.strip(); obj = {"raw": txt}
                meta = {"source_file": str(path), "line_no": i, "json": obj}
                yield txt, meta
    except Exception:
        return

def _yield_txt_chunks(path: Path, para_sep: str = "\n\n"):
    try:
        data = path.read_text(encoding="utf-8", errors="ignore")
        paras = [p.strip() for p in data.split(para_sep) if p.strip()]
        for i, p in enumerate(paras, start=1):
            yield p, {"source_file": str(path), "para_no": i}
    except Exception:
        return

def kb_search(query: str, top_k: int = 8, index_dirs: Optional[list[str]] = None) -> List[Hit]:
    if not query or not query.strip(): return []
    index_dirs = index_dirs or ["data/harvest", "data/miner", "data"]
    q_terms = list(dict.fromkeys(_tokenize(query)))
    if not q_terms: return []
    MAX_FILES = 60
    candidates: List[Hit] = []
    def _score(text: str) -> float:
        tl = text.lower(); L = max(50, len(tl)); hits = 0
        for t in q_terms: hits += min(tl.count(t), 10)
        return hits / (L ** 0.5)
    files = []
    for d in index_dirs:
        p = Path(d)
        if not p.exists(): continue
        files += [Path(x) for x in glob(str(p / "*.jsonl"))]
        files += [Path(x) for x in glob(str(p / "*.txt"))]
    files = files[:MAX_FILES]
    for fp in files:
        gen = _yield_jsonl_chunks(fp) if fp.suffix == ".jsonl" else _yield_txt_chunks(fp)
        for text, meta in gen:
            s = _score(text)
            if s > 0: candidates.append(Hit(text=text, score=s, meta=meta))
    candidates.sort(key=lambda h: h.score, reverse=True)
    return candidates[:top_k]

def kb_fetch(items):
    out = []
    for it in items or []:
        meta = dict(it); src = meta.get("source_file")
        if not src: out.append({"meta": meta, "text": "", "ok": False}); continue
        path = Path(src)
        if not path.exists(): out.append({"meta": meta, "text": "", "ok": False}); continue
        if path.suffix == ".jsonl" and "line_no" in meta:
            try:
                with path.open("r", encoding="utf-8", errors="ignore") as f:
                    for i, line in enumerate(f, start=1):
                        if i == int(meta["line_no"]):
                            try: obj = json.loads(line)
                            except Exception: obj = {"raw": line.strip()}
                            out.append({"meta": meta, "json": obj, "ok": True})
                            break
                    else:
                        out.append({"meta": meta, "text": "", "ok": False})
            except Exception:
                out.append({"meta": meta, "text": "", "ok": False})
        else:
            out.append({"meta": meta, "text": meta.get("text",""), "ok": True})
    return out

def judge_sufficiency(hits, min_hits: int = 3, min_score: float = 0.2, min_chars: int = 500) -> bool:
    if not hits or len(hits) < min_hits: return False
    core = hits[:min_hits]
    avg_score = sum(h.score for h in core) / max(1, len(core))
    total_chars = sum(len(h.text) for h in core)
    return (avg_score >= min_score) and (total_chars >= min_chars)

_CTRL_RX = re.compile(r"[\x00-\x08\x0B\x0C\x0E-\x1F]")
def _safe_text(text: str, max_len: int = 5000) -> str:
    if text is None: return ""
    t = str(text)
    t = _CTRL_RX.sub("", t)
    t = re.sub(r"\s+", " ", t).strip()
    t = html.escape(t, quote=False)
    if len(t) > max_len: t = t[: max_len - 1] + "…"
    return t

_REF_RX = re.compile(r"\[(\d+(?:\s*[-–]\s*\d+)?)\]")
def _extract_used_ref_indexes(answer: str, rationale: str):
    txt = f"{answer or ''}\n{rationale or ''}"
    nums = _REF_RX.findall(txt); out = set()
    for n in nums:
        if "-" in n or "–" in n:
            import re as _re2
            a, b = _re2.split(r"[-–]", n)
            try:
                ai, bi = int(a.strip()), int(b.strip())
                if ai <= bi: out.update(range(ai, bi + 1))
            except ValueError: pass
        else:
            try: out.add(int(n.strip()))
            except ValueError: pass
    return sorted(out)
