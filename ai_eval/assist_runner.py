# ai_eval/assist_runner.py
#!/usr/bin/env python3
import os, sys, json, argparse, time, re, random
from pathlib import Path

def load_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line=line.strip()
            if line:
                yield json.loads(line)

def dump_jsonl(path, rows):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False)+"\n")

def append_jsonl(path, obj):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")

def run_spacy(text, model_path):
    try:
        import spacy
        nlp = spacy.load(model_path)
        doc = nlp(text)
        ents = [{"start": e.start_char, "end": e.end_char, "label": e.label_} for e in getattr(doc, "ents", [])]
        rels = []
        if hasattr(doc._, "relations"):
            rels = list(doc._.relations)
        return {"entities": ents, "relations": rels}
    except Exception:
        return {"entities": [], "relations": []}

PROMPT_SPAN = """You are extracting chemistry entities from a passage.
Labels: ACTION, AMOUNT, ATMOS, CONC, EQUIPMENT, MATERIAL, SPEED, TEMP, TIME, UNIT.
Use the spaCy suggestions as hints but correct them if they are wrong or incomplete.
Return ONLY JSON with this schema:
{"id": <same id>, "text": <original text>, "spans": [{"start":int,"end":int,"label":str}, ...]}
Rules:
- Offsets are 0-based, end-exclusive, relative to the provided text.
- If spaCy suggests an entity that is wrong, drop or fix it.
- If spaCy misses entities, add them.
"""

PROMPT_REL = """You are extracting chemistry entities AND relations.
Entity labels: ACTION, AMOUNT, ATMOS, CONC, EQUIPMENT, MATERIAL, SPEED, TEMP, TIME, UNIT.
Relation labels: MEASURED_IN (AMOUNT→UNIT), HAS_AMOUNT (MATERIAL→AMOUNT) if present.
Use spaCy entities/relations as hints; correct them if needed.
Return ONLY JSON:
{
 "id": <same id>, "text": <original text>,
 "entities":[{"start":int,"end":int,"label":str,"eid":str}, ...],
 "relations":[{"head":str,"tail":str,"label":str}, ...]
}
Rules:
- eids must be unique strings like "e1","e2"... and referenced by relations.
"""

PROMPT_BIO = """You are tagging tokens with BIO (IOB2) scheme for chemistry entities.
Labels: ACTION, AMOUNT, ATMOS, CONC, EQUIPMENT, MATERIAL, SPEED, TEMP, TIME, UNIT.
Given tokens and spaCy's entity hints (converted to token spans), output ONLY JSON:
{"id": <same id>, "tokens":[...], "tags":[...]}
Rules:
- Tags must be same length as tokens.
- Use B-<LABEL>, I-<LABEL>, or O.
"""

PROMPT_STRUCT = """Convert the passage to a structured JSON.
Fields: "procedure" (list of steps), "hardware" (list of equipment strings).
Each step: {"action": str, "material": str|null, "amount": number|null, "unit": str|null}.
Use spaCy as hints; correct them.
Return ONLY JSON:
{"id": <same id>, "output": {"procedure":[...], "hardware":[...] } }
"""

def build_messages(task, ex, hints):
    if task=="span":
        prompt = PROMPT_SPAN
        content = {"id": ex["id"], "text": ex["text"], "spacy_hints": hints["entities"]}
    elif task=="rel":
        prompt = PROMPT_REL
        content = {"id": ex["id"], "text": ex["text"], "spacy_entities": hints["entities"], "spacy_relations": hints["relations"]}
    elif task=="bio":
        prompt = PROMPT_BIO
        content = {"id": ex["id"], "tokens": ex["tokens"], "spacy_hints": hints["entities"]}
    else:
        prompt = PROMPT_STRUCT
        content = {"id": ex["id"], "text": ex["text"], "spacy_hints": hints["entities"]}
    return [
        {"role":"system","content": prompt},
        {"role":"user","content": json.dumps(content, ensure_ascii=False)}
    ]

def parse_json_loose(txt):
    """Robust JSON extraction:
       - strip code fences
       - take between first '{' and last '}'
       - remove trailing commas before } or ]
    """
    if txt is None:
        return None
    s = txt.strip()
    s = re.sub(r"^```(?:json)?|```$", "", s, flags=re.I|re.M).strip()
    i = s.find("{"); j = s.rfind("}")
    if i>=0 and j>i:
        s = s[i:j+1]
    # remove trailing commas like ",}"
    s = re.sub(r",\s*([}\]])", r"\1", s)
    try:
        return json.loads(s)
    except Exception:
        return None

def force_schema(task, ex, obj, fallback_hints):
    """Ensure required keys/types exist; minimal coercion to keep the row usable."""
    if task=="span":
        if not isinstance(obj, dict): return None
        idv = obj.get("id", ex["id"])
        text = obj.get("text", ex["text"])
        spans = obj.get("spans", [])
        good=[]
        if isinstance(spans, list):
            for s in spans:
                if not isinstance(s, dict): continue
                try:
                    st = int(s["start"]); en = int(s["end"]); lab = str(s["label"])
                    if 0 <= st < en <= len(text):
                        good.append({"start":st,"end":en,"label":lab})
                except Exception:
                    continue
        return {"id": idv, "text": text, "spans": good}
    if task=="rel":
        if not isinstance(obj, dict): return None
        idv = obj.get("id", ex["id"])
        text = obj.get("text", ex["text"])
        ents = obj.get("entities", [])
        rels = obj.get("relations", [])
        eout=[]; used=set()
        if isinstance(ents, list):
            k=1
            for e in ents:
                try:
                    st=int(e["start"]); en=int(e["end"]); lab=str(e["label"])
                    eid=str(e.get("eid") or f"e{k}")
                    if eid in used: eid=f"e{k}"
                    used.add(eid); k+=1
                    if 0<=st<en<=len(text):
                        eout.append({"start":st,"end":en,"label":lab,"eid":eid})
                except: continue
        valid_ids = {e["eid"] for e in eout}
        rout=[]
        if isinstance(rels, list):
            for r in rels:
                try:
                    h=str(r["head"]); t=str(r["tail"]); lab=str(r["label"])
                    if h in valid_ids and t in valid_ids:
                        rout.append({"head":h,"tail":t,"label":lab})
                except: continue
        return {"id": idv, "text": text, "entities": eout, "relations": rout}
    if task=="bio":
        if not isinstance(obj, dict): return None
        idv=obj.get("id", ex["id"])
        toks=ex["tokens"]
        tags=obj.get("tags", [])
        if not (isinstance(tags, list) and len(tags)==len(toks)):
            return None
        tags=[str(t) for t in tags]
        return {"id": idv, "tokens": toks, "tags": tags}
    if task=="structured":
        if not isinstance(obj, dict): return None
        idv=obj.get("id", ex["id"])
        out=obj.get("output")
        if not isinstance(out, dict): return None
        return {"id":idv, "output": out}
    return None

def call_openai(messages, model="gpt-4o", temperature=0.0, max_tokens=1200, base_url=None, use_json_mode=True):
    # Try new SDK (openai>=1.0)
    try:
        from openai import OpenAI
        kw={}
        if base_url:
            kw["base_url"]=base_url
        client = OpenAI(**kw)
        req = {"model":model, "messages":messages, "temperature":temperature, "max_tokens":max_tokens}
        if use_json_mode:
            req["response_format"]={"type":"json_object"}
        resp = client.chat.completions.create(**req)
        return resp.choices[0].message.content
    except ModuleNotFoundError:
        # Legacy / Azure fallback
        import openai
        if os.environ.get("AZURE_OPENAI_ENDPOINT"):
            openai.api_type="azure"
            openai.api_base=os.environ["AZURE_OPENAI_ENDPOINT"]
            openai.api_version=os.environ.get("AZURE_OPENAI_API_VERSION","2024-06-01")
            openai.api_key=os.environ.get("AZURE_OPENAI_API_KEY") or os.environ.get("OPENAI_API_KEY")
            # Azure uses deployment name via 'engine'
            resp = openai.ChatCompletion.create(engine=model, messages=messages, temperature=temperature)
            return resp["choices"][0]["message"]["content"]
        else:
            openai.api_key=os.environ["OPENAI_API_KEY"]
            if base_url: openai.base_url=base_url
            resp = openai.ChatCompletion.create(model=model, messages=messages, temperature=temperature)
            return resp["choices"][0]["message"]["content"]

def backoff_sleep(attempt):
    # 0.5, 1, 2, 4, 8...
    time.sleep(min(8, 0.5*(2**attempt) + random.random()*0.2))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True, choices=["span","rel","bio","structured"])
    ap.add_argument("--gold", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--spacy-model", default=None)
    ap.add_argument("--model", default="gpt-4o")
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--max-tokens", type=int, default=1200)
    ap.add_argument("--openai-base", default=None)
    ap.add_argument("--sleep", type=float, default=0.0)
    ap.add_argument("--limit", type=int, default=None, help="Process at most N examples")
    ap.add_argument("--resume", action="store_true", help="Skip IDs already in --out")
    ap.add_argument("--max-chars", type=int, default=2500, help="Truncate text to this many chars before sending")
    ap.add_argument("--retries", type=int, default=3)
    ap.add_argument("--fallback-spacy", action="store_true", help="If GPT JSON fails, emit spaCy-only prediction")
    ap.add_argument("--log-bad", default="runs/assist_bad_json.log")
    args = ap.parse_args()

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    badlog = open(args.log_bad, "a", encoding="utf-8")
    seen = set()

    if args.resume and Path(args.out).exists():
        for row in load_jsonl(args.out):
            seen.add(row.get("id"))

    n_done=0; n_total=0
    for ex in load_jsonl(args.gold):
        ex_id = ex.get("id")
        # Ensure expected inputs exist for each task
        if args.task in ("span","rel","structured"):
            if "text" not in ex or not ex["text"]:
                continue

        if args.resume and ex_id in seen:
            continue

        n_total += 1
        if args.limit and n_total > args.limit:
            break

        # Truncate long text to keep token usage sane
        if args.task in ("span","rel","structured") and args.max_chars and len(ex["text"]) > args.max_chars:
            ex = dict(ex)
            ex["text"] = ex["text"][:args.max_chars]

        hints = {"entities": [], "relations": []}
        if args.spacy_model and args.task in ("span","rel","structured") and ex.get("text"):
            hints = run_spacy(ex["text"], args.spacy_model)

        messages = build_messages(args.task, ex, hints)

        # Try strict JSON mode first, then a no-JSON-mode retry
        txt=None; obj=None
        for attempt in range(args.retries):
            try:
                use_json_mode = (attempt==0)  # first try in JSON mode
                txt = call_openai(messages, model=args.model, temperature=args.temperature,
                                  max_tokens=args.max_tokens, base_url=args.openai_base,
                                  use_json_mode=use_json_mode)
                obj = parse_json_loose(txt)
                if obj: break
            except Exception as e:
                badlog.write(f"[ERROR] id={ex_id} attempt={attempt} error={e}\n")
            backoff_sleep(attempt)

        if not obj:
            # One minimal reprompt (non-JSON mode) to coerce raw JSON
            try:
                retry_msgs = messages + [{"role":"system","content":"Return ONLY VALID JSON for the required schema. No commentary, no code fences."}]
                txt = call_openai(retry_msgs, model=args.model, temperature=args.temperature,
                                  max_tokens=args.max_tokens, base_url=args.openai_base,
                                  use_json_mode=False)
                obj = parse_json_loose(txt)
            except Exception as e:
                badlog.write(f"[ERROR] id={ex_id} final-retry error={e}\n")

        row=None
        if obj:
            row = force_schema(args.task, ex, obj, hints)

        if row is None and args.fallback_spacy and args.spacy_model:
            # Emit spaCy-only fallback so you don't lose the example
            if args.task=="span":
                row = {"id": ex["id"], "text": ex["text"], "spans": hints["entities"]}
            elif args.task=="rel":
                ents=[]; used=set(); k=1
                for e in hints["entities"]:
                    try:
                        st=int(e["start"]); en=int(e["end"]); lab=str(e["label"])
                        eid=f"e{k}"; k+=1
                        if 0<=st<en<=len(ex["text"]):
                            ents.append({"start":st,"end":en,"label":lab,"eid":eid})
                    except: continue
                row = {"id": ex["id"], "text": ex["text"], "entities": ents, "relations": hints["relations"]}
            elif args.task=="structured":
                row = {"id": ex["id"], "output": {"procedure":[], "hardware":[]}}
            # (BIO fallback skipped—needs tokens)
        if row is None:
            badlog.write(f"[WARN] JSON parse failed for id={ex_id}; skipping\n")
        else:
            append_jsonl(args.out, row)
            n_done += 1

        if args.sleep>0:
            time.sleep(args.sleep)

    badlog.close()
    print(f"Wrote {n_done} predictions to {args.out} (processed={n_total}, skipped_seen={len(seen)})")

if __name__ == "__main__":
    import re
    main()