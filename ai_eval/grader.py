import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from typing import Any, Dict, List, Tuple


# ---------- IO ----------
def load_cfg(path: str) -> Dict[str, Any]:
    try:
        import yaml

        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    except Exception:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)


def load_jsonl(path: str) -> List[Dict[str, Any]]:
    out = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            out.append(json.loads(line))
    return out


def save_json(path: str, obj: Any):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


# ---------- Metrics ----------
def prf1(tp, fp, fn):
    p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
    return p, r, f1


def brier_score(probs: List[float], golds: List[int]):
    # probs in [0,1], gold in {0,1}
    n = max(1, len(probs))
    return sum((p - y) ** 2 for p, y in zip(probs, golds)) / n


def ece(probs: List[float], golds: List[int], n_bins: int = 10):
    if not probs:
        return 0.0
    bins = [[] for _ in range(n_bins)]
    for p, y in zip(probs, golds):
        b = min(n_bins - 1, int(p * n_bins))
        bins[b].append((p, y))
    err = 0.0
    total = 0
    for b in bins:
        if not b:
            continue
        conf = sum(p for p, _ in b) / len(b)
        acc = sum(1 for _, y in b if y == 1) / len(b)
        err += len(b) * abs(conf - acc)
        total += len(b)
    return err / max(1, total)


# ---------- Common helpers ----------
def iou(a: Tuple[int, int], b: Tuple[int, int]) -> float:
    a0, a1 = a
    b0, b1 = b
    if a0 > a1:
        a0, a1 = a1, a0
    if b0 > b1:
        b0, b1 = b1, b0
    inter = max(0, min(a1, b1) - max(a0, b0))
    union = max(a1, a0) - min(a0, a1) + max(b1, b0) - min(b0, b1) - inter
    return (inter / union) if union > 0 else 0.0


def simple_tokens_with_spans(text: str):
    toks = []
    i = 0
    while i < len(text):
        if text[i].isspace():
            i += 1
            continue
        j = i
        while j < len(text) and not text[j].isspace():
            j += 1
        toks.append((text[i:j], i, j))
        i = j
    return toks


def char_to_token_idx(toks, start, end):
    for idx, (_, s, e) in enumerate(toks):
        if not (e <= start or s >= end):
            return idx
    center = (start + end) // 2
    best, bestd = 0, 10**9
    for idx, (_, s, e) in enumerate(toks):
        d = min(abs(center - s), abs(center - e))
        if d < bestd:
            bestd = d
            best = idx
    return best


def normalize_surface(s: str, casefold: bool, strip_punct: bool) -> str:
    if s is None:
        return ""
    if casefold:
        s = s.casefold()
    if strip_punct:
        s = s.strip(" ,.;:!?()[]{}\"'`")
    return s


# ---------- Span NER ----------
def safe_spans(example: Dict[str, Any]) -> List[Dict[str, Any]]:
    out = []
    for s in example.get("spans", []):
        if all(k in s for k in ("start", "end", "label")):
            out.append(
                {
                    "start": int(s["start"]),
                    "end": int(s["end"]),
                    "label": str(s["label"]),
                    **{
                        k: v for k, v in s.items() if k not in ("start", "end", "label")
                    },
                }
            )
    return out


def align_spans(
    gspans, pspans, text, ptext, mode="iou", thr=0.5, casefold=True, strip_punct=False
):
    # per-label greedy
    by_lbl_g = defaultdict(list)
    by_lbl_p = defaultdict(list)
    for i, g in enumerate(gspans):
        by_lbl_g[g["label"]].append((i, g))
    for j, p in enumerate(pspans):
        by_lbl_p[p["label"]].append((j, p))
    matches = []
    for lbl in set(list(by_lbl_g.keys()) + list(by_lbl_p.keys())):
        glist = by_lbl_g.get(lbl, [])
        plist = by_lbl_p.get(lbl, [])
        cand = []
        for gi, g in glist:
            gs = (g["start"], g["end"])
            gs_text = text[g["start"] : g["end"]] if text else None
            for pj, p in plist:
                ps = (p["start"], p["end"])
                score = (
                    1.0
                    if (gs == ps and mode == "exact")
                    else (iou(gs, ps) if mode != "exact" else 0.0)
                )
                if score < (1.0 if mode == "exact" else thr):
                    continue
                if text and ptext:
                    if normalize_surface(
                        gs_text, casefold, strip_punct
                    ) == normalize_surface(
                        ptext[p["start"] : p["end"]], casefold, strip_punct
                    ):
                        score += 1e-6
                cand.append((score, gi, pj))
        cand.sort(reverse=True)
        used_g = set()
        used_p = set()
        for sc, gi, pj in cand:
            if gi in used_g or pj in used_p:
                continue
            matches.append((gi, pj, sc))
            used_g.add(gi)
            used_p.add(pj)
    matched_g = {gi for gi, _, _ in matches}
    matched_p = {pj for _, pj, _ in matches}
    return matches, matched_g, matched_p


# ---------- Attribute rules ----------
def enforce_attribute_rules(
    rules, text, gspans, pspans, matches, per_label, attr_report
):
    if not rules:
        return
    toks = simple_tokens_with_spans(text or "")
    gold_by_idx = {i: s for i, s in enumerate(gspans)}
    pred_by_idx = {j: s for j, s in enumerate(pspans)}
    g2p = {gi: pj for gi, pj, _ in matches}

    # quick lookup lists per label
    gold_by_lbl = defaultdict(list)
    pred_by_lbl = defaultdict(list)
    for s in gspans:
        gold_by_lbl[s["label"]].append(s)
    for s in pspans:
        pred_by_lbl[s["label"]].append(s)

    def has_neighbor(span, neighbor_spans, win_chars=None, win_tokens=None):
        if win_tokens is not None:
            h = char_to_token_idx(toks, span["start"], span["end"])
            for nb in neighbor_spans:
                n = char_to_token_idx(toks, nb["start"], nb["end"])
                if abs(h - n) <= win_tokens:
                    return True
            return False
        else:

            def dist(a, b):
                if a["end"] <= b["start"]:
                    return b["start"] - a["end"]
                if b["end"] <= a["start"]:
                    return a["start"] - b["end"]
                return 0

            for nb in neighbor_spans:
                if win_chars is None or dist(span, nb) <= win_chars:
                    return True
            return False

    def regex_ok(span, pattern, text):
        if not pattern:
            return True
        s = text[span["start"] : span["end"]] if text else ""
        return re.fullmatch(pattern, s) is not None

    def numeric_in_range(span, rng, text):
        if not rng:
            return True
        s = text[span["start"] : span["end"]] if text else ""
        m = re.search(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", s)
        if not m:
            return False
        val = float(m.group(0))
        lo = rng.get("min", -1e18)
        hi = rng.get("max", 1e18)
        return lo <= val <= hi

    for r in rules:
        rtype = r.get("type")
        if rtype == "cooccurrence":
            head = r["head_label"]
            neighbor = r["neighbor_label"]
            win = r.get("window", {})
            wchars = win.get("chars")
            wtok = win.get("tokens")
            # only apply where gold head truly needs neighbor (to avoid unfair penalties)
            gold_heads = [
                (gi, gold_by_idx[gi])
                for gi in range(len(gspans))
                if gspans[gi]["label"] == head
            ]
            gold_nei = [s for s in gspans if s["label"] == neighbor]
            requiring = [
                (gi, s)
                for gi, s in gold_heads
                if has_neighbor(s, gold_nei, wchars, wtok)
            ]
            pred_nei = [s for s in pspans if s["label"] == neighbor]
            for gi, gspan in requiring:
                pj = g2p.get(gi)
                if pj is None:
                    continue  # already FN
                pspan = pred_by_idx[pj]
                ok = has_neighbor(pspan, pred_nei, wchars, wtok)
                if not ok:
                    per_label[head]["tp"] -= 1
                    per_label[head]["fn"] += 1
                    per_label[head]["fp"] += 1
                    attr_report["demoted"].append(
                        {
                            "rule": "cooccurrence",
                            "head": head,
                            "neighbor": neighbor,
                            "example_text": text[
                                max(0, gspan["start"] - 30) : gspan["end"] + 30
                            ],
                        }
                    )
        elif rtype == "regex":
            lbl = r["label"]
            pattern = r.get("pattern")
            # apply to matched preds for this label
            for gi, pj, _ in matches:
                if gspans[gi]["label"] != lbl:
                    continue
                pspan = pred_by_idx[pj]
                if not regex_ok(pspan, pattern, text):
                    per_label[lbl]["tp"] -= 1
                    per_label[lbl]["fn"] += 1
                    per_label[lbl]["fp"] += 1
                    attr_report["demoted"].append(
                        {
                            "rule": "regex",
                            "label": lbl,
                            "pattern": pattern,
                            "surface": text[pspan["start"] : pspan["end"]],
                        }
                    )
        elif rtype == "numeric_range":
            lbl = r["label"]
            rng = r.get("range", {})
            for gi, pj, _ in matches:
                if gspans[gi]["label"] != lbl:
                    continue
                pspan = pred_by_idx[pj]
                if not numeric_in_range(pspan, rng, text):
                    per_label[lbl]["tp"] -= 1
                    per_label[lbl]["fn"] += 1
                    per_label[lbl]["fp"] += 1
                    attr_report["demoted"].append(
                        {
                            "rule": "numeric_range",
                            "label": lbl,
                            "range": rng,
                            "surface": text[pspan["start"] : pspan["end"]],
                        }
                    )
        elif rtype == "typed_unit":
            # e.g., TEMP must have unit in set; SPEED must end with rpm; CONC units allowed set
            lbl = r["label"]
            units = set(r.get("units", []))
            mode = r.get("mode", "suffix")  # suffix|any
            pat_any = r.get("pattern")  # optional regex override
            for gi, pj, _ in matches:
                if gspans[gi]["label"] != lbl:
                    continue
                surf = text[pspans[pj]["start"] : pspans[pj]["end"]]
                ok = True
                if pat_any:
                    ok = re.search(pat_any, surf) is not None
                else:
                    if mode == "suffix":
                        ok = any(surf.strip().endswith(u) for u in units)
                    else:
                        ok = any(u in surf for u in units)
                if not ok:
                    per_label[lbl]["tp"] -= 1
                    per_label[lbl]["fn"] += 1
                    per_label[lbl]["fp"] += 1
                    attr_report["demoted"].append(
                        {
                            "rule": "typed_unit",
                            "label": lbl,
                            "units": sorted(list(units)),
                            "surface": surf,
                        }
                    )


# ---------- BIO ----------
def bio_eval(
    gold, preds, labels: List[str], ignore_o=True, report_entities=True, probs_key=None
):
    pred_by_id = {ex["id"]: ex for ex in preds}
    per_label_tok = defaultdict(lambda: Counter(tp=0, fp=0, fn=0))
    support_tok = Counter()
    prob_pos = []
    gold_pos = []

    def tlabel(tag):
        if tag == "O" or tag is None:
            return None
        return tag.split("-", 1)[1] if "-" in tag else tag

    def bio_to_spans(tags):
        spans = []
        i = 0
        while i < len(tags):
            t = tags[i]
            if t.startswith("B-"):
                lbl = t.split("-", 1)[1]
                j = i + 1
                while j < len(tags) and tags[j] == f"I-{lbl}":
                    j += 1
                spans.append((i, j, lbl))
                i = j
            else:
                i += 1
        return spans

    for gex in gold:
        gid = gex["id"]
        pex = pred_by_id.get(
            gid, {"tokens": gex["tokens"], "tags": ["O"] * len(gex["tokens"])}
        )
        gtags = list(gex["tags"])
        ptags = list(pex["tags"])
        n = min(len(gtags), len(ptags))
        gtags = gtags[:n]
        ptags = ptags[:n]
        for k in range(n):
            gl = tlabel(gtags[k])
            pl = tlabel(ptags[k])
            if ignore_o and (gl is None and pl is None):
                continue
            if gl is not None:
                support_tok[gl] += 1
            if gl is not None and pl == gl:
                per_label_tok[gl]["tp"] += 1
            if gl is not None and pl != gl:
                per_label_tok[gl]["fn"] += 1
            if pl is not None and pl != gl:
                per_label_tok[pl]["fp"] += 1
        # collect optional token probs for calibration (expects probs per token for positive class of predicted label)
        if probs_key and probs_key in pex:
            probs = pex[probs_key][:n]
            for p, gt, pt in zip(probs, gtags, ptags):
                is_correct = int(gt == pt and gt != "O")
                prob_pos.append(float(p))
                gold_pos.append(is_correct)

    tok_summary = []
    micro = Counter(tp=0, fp=0, fn=0)
    for lbl, c in sorted(per_label_tok.items()):
        micro["tp"] += c["tp"]
        micro["fp"] += c["fp"]
        micro["fn"] += c["fn"]
        p, r, f1 = prf1(c["tp"], c["fp"], c["fn"])
        tok_summary.append(
            {
                "label": lbl,
                "support_tokens": int(support_tok[lbl]),
                "precision": p,
                "recall": r,
                "f1": f1,
            }
        )
    mp, mr, mf1 = prf1(micro["tp"], micro["fp"], micro["fn"])
    token_report = {
        "micro": {
            "precision": mp,
            "recall": mr,
            "f1": mf1,
            "tp": micro["tp"],
            "fp": micro["fp"],
            "fn": micro["fn"],
        },
        "per_label": tok_summary,
    }
    calib = None
    if prob_pos:
        calib = {
            "brier": brier_score(prob_pos, gold_pos),
            "ece": ece(prob_pos, gold_pos, 10),
        }
    ent_report = None
    if report_entities:
        # exact entity match on token spans
        per_label_ent = defaultdict(lambda: Counter(tp=0, fp=0, fn=0))
        support_ent = Counter()
        pred_by_id = {ex["id"]: ex for ex in preds}
        for gex in gold:
            gid = gex["id"]
            pex = pred_by_id.get(
                gid, {"tokens": gex["tokens"], "tags": ["O"] * len(gex["tokens"])}
            )
            g_sp = bio_to_spans(gex["tags"])
            p_sp = bio_to_spans(pex["tags"])
            g_by = defaultdict(list)
            p_by = defaultdict(list)
            for s in g_sp:
                g_by[s[2]].append(s)
            for s in p_sp:
                p_by[s[2]].append(s)
            for lbl in set(list(g_by.keys()) + list(p_by.keys())):
                G = g_by.get(lbl, [])
                P = p_by.get(lbl, [])
                used = set()
                for gs in G:
                    support_ent[lbl] += 1
                    matched = False
                    for j, ps in enumerate(P):
                        if j in used:
                            continue
                        if gs[0] == ps[0] and gs[1] == ps[1]:
                            per_label_ent[lbl]["tp"] += 1
                            used.add(j)
                            matched = True
                            break
                    if not matched:
                        per_label_ent[lbl]["fn"] += 1
                per_label_ent[lbl]["fp"] += max(0, len(P) - len(used))
        ent_summary = []
        micro2 = Counter(tp=0, fp=0, fn=0)
        for lbl, c in sorted(per_label_ent.items()):
            micro2["tp"] += c["tp"]
            micro2["fp"] += c["fp"]
            micro2["fn"] += c["fn"]
            p, r, f1 = prf1(c["tp"], c["fp"], c["fn"])
            ent_summary.append(
                {
                    "label": lbl,
                    "support_entities": int(support_ent[lbl]),
                    "precision": p,
                    "recall": r,
                    "f1": f1,
                }
            )
        mp2, mr2, mf12 = prf1(micro2["tp"], micro2["fp"], micro2["fn"])
        ent_report = {
            "micro": {
                "precision": mp2,
                "recall": mr2,
                "f1": mf12,
                "tp": micro2["tp"],
                "fp": micro2["fp"],
                "fn": micro2["fn"],
            },
            "per_label": ent_summary,
        }

    return token_report, ent_report, calib


# ---------- Relations ----------
def safe_entities(ex):
    ents = []
    for e in ex.get("entities", []):
        if all(k in e for k in ("start", "end", "label")):
            eid = e.get("eid")
            ents.append(
                {
                    "start": int(e["start"]),
                    "end": int(e["end"]),
                    "label": str(e["label"]),
                    "eid": eid,
                }
            )
    return ents


def safe_relations(ex):
    rels = []
    for r in ex.get("relations", []):
        if all(k in r for k in ("head", "tail", "label")):
            rels.append(
                {"head": r["head"], "tail": r["tail"], "label": str(r["label"])}
            )
    return rels


def rel_eval(gold, preds, match_mode="iou", thr=0.5):
    pred_by_id = {ex["id"]: ex for ex in preds}
    per_label = defaultdict(lambda: Counter(tp=0, fp=0, fn=0))
    support = Counter()
    errors = []

    for gex in gold:
        gid = gex["id"]
        pex = pred_by_id.get(gid, {})
        text = gex.get("text")
        pex.get("text", text)
        gents = safe_entities(gex)
        pents = safe_entities(pex)
        grels = safe_relations(gex)
        prels = safe_relations(pex)

        # entity alignment across ANY label (so we can catch label confusions too)
        # build best match table by IoU; we will allow cross-label in alignment but relation credit requires labels match
        cand = []
        for gi, ge in enumerate(gents):
            gs = (ge["start"], ge["end"])
            for pj, pe in enumerate(pents):
                ps = (pe["start"], pe["end"])
                score = (
                    1.0
                    if (gs == ps and match_mode == "exact")
                    else (iou(gs, ps) if match_mode != "exact" else 0.0)
                )
                if score < (1.0 if match_mode == "exact" else thr):
                    continue
                cand.append((score, gi, pj))
        cand.sort(reverse=True)
        used_g = set()
        used_p = set()
        align = {}
        for sc, gi, pj in cand:
            if gi in used_g or pj in used_p:
                continue
            align[gi] = pj
            used_g.add(gi)
            used_p.add(pj)

        # index relations by (gi,gj,lbl)
        g_by = defaultdict(list)
        for r in grels:
            # map eids to indices
            def idx_from_ref(ref):
                if isinstance(ref, int):
                    return ref
                if isinstance(ref, str) and ref.startswith("e"):
                    for i, e in enumerate(gents):
                        if e.get("eid") == ref:
                            return i
                # fallback: try int
                try:
                    return int(ref)
                except Exception:
                    return None

            hi = idx_from_ref(r["head"])
            ti = idx_from_ref(r["tail"])
            if hi is None or ti is None:
                continue
            support[r["label"]] += 1
            g_by[r["label"]].append((hi, ti))

        # predicted relations mapped through alignment
        p_by = defaultdict(list)
        for r in prels:

            def pidx(ref):
                if isinstance(ref, int):
                    return ref
                if isinstance(ref, str) and ref.startswith("e"):
                    for j, e in enumerate(pents):
                        if e.get("eid") == ref:
                            return j
                try:
                    return int(ref)
                except Exception:
                    return None

            hj = pidx(r["head"])
            tj = pidx(r["tail"])
            if hj is None or tj is None:
                continue
            p_by[r["label"]].append((hj, tj))

        # count TP/FN/FP per label using alignment
        for lbl in set(list(g_by.keys()) + list(p_by.keys())):
            gold_pairs = set()
            for hi, ti in g_by.get(lbl, []):
                # require heads/tails label to match too? make it stricter via entity labels check
                gold_pairs.add((hi, ti))
            pred_pairs = set()
            for hj, tj in p_by.get(lbl, []):
                pred_pairs.add((hj, tj))

            # Map pred entity indices back to gold via reverse alignment for direct comparison
            pred_pairs_mapped = set()
            rev = {v: k for k, v in align.items()}
            for hj, tj in pred_pairs:
                if hj in rev and tj in rev:
                    pred_pairs_mapped.add((rev[hj], rev[tj]))

            tp_pairs = gold_pairs & pred_pairs_mapped
            fn_pairs = gold_pairs - pred_pairs_mapped
            fp_pairs = pred_pairs_mapped - gold_pairs

            per_label[lbl]["tp"] += len(tp_pairs)
            per_label[lbl]["fn"] += len(fn_pairs)
            per_label[lbl]["fp"] += len(fp_pairs)

            # error dump
            for hi, ti in fn_pairs:
                h = gents[hi]
                t = gents[ti]
                errors.append(
                    {
                        "id": gid,
                        "type": "FN",
                        "label": lbl,
                        "head_surface": text[h["start"] : h["end"]],
                        "tail_surface": text[t["start"] : t["end"]],
                    }
                )

            for hi, ti in fp_pairs:
                h = gents[hi]
                t = gents[ti]
                errors.append(
                    {
                        "id": gid,
                        "type": "FP",
                        "label": lbl,
                        "head_surface": text[h["start"] : h["end"]],
                        "tail_surface": text[t["start"] : t["end"]],
                    }
                )

    # aggregate
    summary = []
    micro = Counter(tp=0, fp=0, fn=0)
    for lbl, c in sorted(per_label.items()):
        micro["tp"] += c["tp"]
        micro["fp"] += c["fp"]
        micro["fn"] += c["fn"]
        p, r, f1 = prf1(c["tp"], c["fp"], c["fn"])
        summary.append(
            {
                "label": lbl,
                "support": int(support[lbl]),
                "precision": p,
                "recall": r,
                "f1": f1,
            }
        )
    mp, mr, mf1 = prf1(micro["tp"], micro["fp"], micro["fn"])
    return {
        "micro": {
            "precision": mp,
            "recall": mr,
            "f1": mf1,
            "tp": micro["tp"],
            "fp": micro["fp"],
            "fn": micro["fn"],
        },
        "per_label": summary,
        "errors": errors,
    }


# ---------- Structured output ----------
def struct_eval(gold, preds, schema=None):
    """Validate JSON outputs against a minimal schema and compare to gold (presence + normalized values)."""
    pred_by_id = {ex["id"]: ex for ex in preds}
    try:
        import jsonschema  # type: ignore

        use_jsonschema = True
    except Exception:
        use_jsonschema = False

    valid = 0
    total = 0
    errors = []
    for g in gold:
        gid = g["id"]
        total += 1
        out = pred_by_id.get(gid, {}).get("output")
        if out is None:
            errors.append({"id": gid, "error": "missing_output"})
            continue
        if use_jsonschema and isinstance(schema, dict) and "$schema" in schema:
            try:
                jsonschema.validate(out, schema)
                valid += 1
            except Exception as e:
                errors.append({"id": gid, "error": "jsonschema_fail", "detail": str(e)})
        else:
            # minimal checks
            req = (schema or {}).get("required", [])
            types = (schema or {}).get("types", {})
            ok = True
            for k in req:
                if k not in out:
                    ok = False
                    errors.append({"id": gid, "error": "missing_key", "key": k})
            for k, t in types.items():
                if k in out and t == "list" and not isinstance(out[k], list):
                    ok = False
                    errors.append(
                        {"id": gid, "error": "type_error", "key": k, "expect": "list"}
                    )
                if k in out and t == "object" and not isinstance(out[k], dict):
                    ok = False
                    errors.append(
                        {"id": gid, "error": "type_error", "key": k, "expect": "object"}
                    )
            if ok:
                valid += 1
    return {"valid_rate": valid / max(1, total), "total": total, "errors": errors}


# ---------- Slices ----------
def make_slices(cfg_slices):
    """cfg_slices: list of {name:..., pattern:...} applied to 'text' field of gold example."""
    out = []
    for s in cfg_slices or []:
        try:
            out.append((s["name"], re.compile(s["pattern"], re.I)))
        except Exception:
            pass
    return out


def filter_indices_by_slice(gold, rx):
    idxs = []
    for i, g in enumerate(gold):
        if rx.search(g.get("text", "") or ""):
            idxs.append(i)
    return idxs


# ---------- Main runners ----------
def run_span(cfg):
    gold = load_jsonl(cfg["dataset"]["gold_path"])
    preds = load_jsonl(cfg["predictions"]["pred_path"])
    labels = set(cfg.get("labels", []))
    match = cfg.get("matching", {})
    mode = match.get("mode", "iou")
    thr = float(match.get("iou_threshold", 0.5))
    casefold = bool(match.get("casefold", True))
    strip_punct = bool(match.get("strip_punct", False))
    rules = cfg.get("attribute_rules", [])
    out_path = cfg.get("report", {}).get("out_path", "reports/report_span.json")
    errors_dump = cfg.get("report", {}).get("errors_path", "reports/errors_span.jsonl")

    per_label = defaultdict(lambda: Counter(tp=0, fp=0, fn=0))
    support = Counter()
    attr_report = {"demoted": []}
    all_errors = []

    pred_map = {ex["id"]: ex for ex in preds}

    for gex in gold:
        gid = gex["id"]
        pex = pred_map.get(gid, {"spans": []})
        gsp = safe_spans(gex)
        psp = safe_spans(pex)
        if labels:
            gsp = [s for s in gsp if s["label"] in labels]
            psp = [s for s in psp if s["label"] in labels]
        for s in gsp:
            support[s["label"]] += 1
        matches, mg, mp = align_spans(
            gsp,
            psp,
            gex.get("text"),
            pex.get("text", gex.get("text")),
            mode,
            thr,
            casefold,
            strip_punct,
        )
        for gi, pj, _ in matches:
            per_label[gsp[gi]["label"]]["tp"] += 1
        for gi, s in enumerate(gsp):
            if gi not in mg:
                per_label[s["label"]]["fn"] += 1
        for pj, s in enumerate(psp):
            if pj not in mp:
                per_label[s["label"]]["fp"] += 1

        enforce_attribute_rules(
            rules, gex.get("text", ""), gsp, psp, matches, per_label, attr_report
        )

        # error dump
        for gi, s in enumerate(gsp):
            if gi not in mg:
                ctx = gex.get("text", "")[max(0, s["start"] - 30) : s["end"] + 30]
                all_errors.append(
                    {"id": gid, "type": "FN", "label": s["label"], "context": ctx}
                )
        for pj, s in enumerate(psp):
            if pj not in mp:
                ctx = gex.get("text", "")[max(0, s["start"] - 30) : s["end"] + 30]
                all_errors.append(
                    {"id": gid, "type": "FP", "label": s["label"], "context": ctx}
                )

    rows = []
    micro = Counter(tp=0, fp=0, fn=0)
    for lbl, c in sorted(per_label.items()):
        micro["tp"] += c["tp"]
        micro["fp"] += c["fp"]
        micro["fn"] += c["fn"]
        p, r, f1 = prf1(c["tp"], c["fp"], c["fn"])
        rows.append(
            {
                "label": lbl,
                "support": int(support[lbl]),
                "precision": p,
                "recall": r,
                "f1": f1,
            }
        )
    mp, mr, mf1 = prf1(micro["tp"], micro["fp"], micro["fn"])

    report = {
        "task": "entity_extraction",
        "micro": {
            "precision": mp,
            "recall": mr,
            "f1": mf1,
            "tp": micro["tp"],
            "fp": micro["fp"],
            "fn": micro["fn"],
        },
        "per_label": rows,
        "support": dict(support),
        "attribute_demoted_tp": len(attr_report["demoted"]),
    }
    save_json(out_path, report)
    with open(errors_dump, "w", encoding="utf-8") as f:
        for e in all_errors:
            f.write(json.dumps(e, ensure_ascii=False) + "\n")

    # slices
    slices = make_slices(cfg.get("slices"))
    if slices:
        slice_out = {}
        for name, rx in slices:
            idxs = filter_indices_by_slice(gold, rx)
            # compute micro-F1 restricted to these indices (approx: recompute per-label locally)
            pl = defaultdict(lambda: Counter(tp=0, fp=0, fn=0))
            sup = Counter()
            for i in idxs:
                gex = gold[i]
                pex = pred_map.get(gex["id"], {"spans": []})
                gsp = safe_spans(gex)
                psp = safe_spans(pex)
                if labels:
                    gsp = [s for s in gsp if s["label"] in labels]
                    psp = [s for s in psp if s["label"] in labels]
                for s in gsp:
                    sup[s["label"]] += 1
                matches, mg, mp = align_spans(
                    gsp,
                    psp,
                    gex.get("text"),
                    pex.get("text", gex.get("text")),
                    mode,
                    thr,
                    casefold,
                    strip_punct,
                )
                for gi, pj, _ in matches:
                    pl[gsp[gi]["label"]]["tp"] += 1
                for gi, s in enumerate(gsp):
                    if gi not in mg:
                        pl[s["label"]]["fn"] += 1
                for pj, s in enumerate(psp):
                    if pj not in mp:
                        pl[s["label"]]["fp"] += 1
            micro = Counter(tp=0, fp=0, fn=0)
            for lbl, c in pl.items():
                micro["tp"] += c["tp"]
                micro["fp"] += c["fp"]
                micro["fn"] += c["fn"]
            mp, mr, mf1 = prf1(micro["tp"], micro["fp"], micro["fn"])
            slice_out[name] = {
                "micro": {"precision": mp, "recall": mr, "f1": mf1},
                "support": dict(sup),
            }
        save_json(out_path.replace(".json", "_slices.json"), slice_out)

    print(f"Span eval → P={mp:.3f} R={mr:.3f} F1={mf1:.3f}. Report: {out_path}")


def run_bio(cfg):
    gold = load_jsonl(cfg["dataset"]["gold_path"])
    preds = load_jsonl(cfg["predictions"]["pred_path"])
    labels = set(cfg.get("labels", []))
    out_path = cfg.get("report", {}).get("out_path", "reports/report_bio.json")
    ignore_o = bool(cfg.get("bio", {}).get("ignore_o", True))
    report_entities = bool(cfg.get("bio", {}).get("report_entities", True))
    probs_key = cfg.get("bio", {}).get("probs_key")  # optional

    tok, ent, calib = bio_eval(
        gold, preds, list(labels), ignore_o, report_entities, probs_key
    )
    report = {"task": "bio_token", "token_level": tok}
    if ent:
        report["entity_level"] = ent
    if calib:
        report["calibration"] = calib
    save_json(out_path, report)
    print(f"BIO eval → token micro F1={tok['micro']['f1']:.3f}. Report: {out_path}")


def run_rel(cfg):
    gold = load_jsonl(cfg["dataset"]["gold_path"])
    preds = load_jsonl(cfg["predictions"]["pred_path"])
    match = cfg.get("matching", {})
    mode = match.get("mode", "iou")
    thr = float(match.get("iou_threshold", 0.5))
    out_path = cfg.get("report", {}).get("out_path", "reports/report_rel.json")
    rep = rel_eval(gold, preds, mode, thr)
    save_json(out_path, rep)
    # dump errors
    err_path = cfg.get("report", {}).get("errors_path", "reports/errors_rel.jsonl")
    with open(err_path, "w", encoding="utf-8") as f:
        for e in rep["errors"]:
            f.write(json.dumps(e, ensure_ascii=False) + "\n")
    print(f"Relation eval → micro F1={rep['micro']['f1']:.3f}. Report: {out_path}")


def run_struct(cfg):
    gold = load_jsonl(cfg["dataset"]["gold_path"])
    preds = load_jsonl(cfg["predictions"]["pred_path"])
    out_path = cfg.get("report", {}).get("out_path", "reports/report_struct.json")
    schema = cfg.get("schema")  # dict or JSON path
    if isinstance(schema, str) and schema.endswith((".json", ".schema")):
        try:
            with open(schema, "r", encoding="utf-8") as f:
                schema = json.load(f)
        except Exception:
            schema = None
    rep = struct_eval(gold, preds, schema=schema)
    save_json(out_path, rep)
    print(
        f"Structured-output eval → valid rate={rep['valid_rate']:.3f}. Report: {out_path}"
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-c", "--config", required=True, help="Path to YAML/JSON config")
    args = ap.parse_args()
    cfg = load_cfg(args.config)
    task = cfg.get("task", "entity_extraction")
    if task == "entity_extraction":
        return run_span(cfg)
    if task == "bio_token":
        return run_bio(cfg)
    if task == "relations":
        return run_rel(cfg)
    if task == "structured_output":
        return run_struct(cfg)
    print("Unknown task:", task)
    sys.exit(2)


if __name__ == "__main__":
    main()
