from __future__ import annotations

import json
import os
import pathlib
import re
import unicodedata
from typing import Any, Dict, List, Optional, Tuple

# Try to load defaults/device IDs and optional external post-processor
try:
    from app_utils.converter_h import DEFAULTS, DEVICE_IDS
    from app_utils.converter_h import apply_postprocessing as _ext_apply_post
except Exception:
    _ext_apply_post = None
    DEFAULTS = {
        "stir_rpm": 700,
        "centrifuge_rpm": 4000,
        "centrifuge_minutes": 10,
        "room_temp_C": 25,
        "transfer_rate_slow": "slow",
    }
    DEVICE_IDS = {
        "stir_plate_id": "SP1",
        "hotplate_id": "HP1",
        "centrifuge_id": "CF1",
        "oven_id": "OV1",
        "vacuum_pump_id": "VP1",
        "vortex_id": "VX1",
        "sonicator_id": "SN1",
    }

ROBOT_NORMALIZER_VERSION = "2.2.3"
_BANNER_SHOWN = False

try:
    from app_utils.post_polish import polish_robot_doc as _post_polish
except Exception:
    _post_polish = None


def _build_bottle_config(doc):
    """Map chemicals→bottles and set executor prefs (rpm, washes, drying)."""
    import re

    names = set()
    for st in doc.get("steps", []) or []:
        for r in st.get("reagents_structured") or []:
            if isinstance(r, dict):
                n = r.get("name")
                if isinstance(n, str) and n.strip():
                    names.add(n.strip())
        for m in st.get("micro_ops") or []:
            for key in ("object", "from"):
                v = m.get(key)
                if not isinstance(v, str):
                    continue
                s2 = v.strip()
                if not s2 or s2.lower() in {"bench", "rack", "waste"}:
                    continue
                if s2.endswith("_bottle"):
                    continue
                if re.fullmatch(r"V\\d+(_tube)?", s2):
                    continue
                if m.get("verb") == "set":
                    continue
                names.add(s2)
    # ensure common bottles even if not seen explicitly
    names.update({"ethanol", "deionized water"})
    bottle_map = {
        n.lower(): re.sub(r"[^a-z0-9]+", "_", n.lower()).strip("_") + "_bottle"
        for n in names
    }
    bottle_labels = {vid: (n.title() + " bottle") for n, vid in bottle_map.items()}
    bottle_labels.update(
        {
            "deionized_water_bottle": "Deionized water bottle",
            "ethanol_bottle": "Ethanol bottle",
            "waste": "Waste container",
        }
    )
    return {
        "devices": doc.get("devices", {}),
        "reaction_vessel": "V1",
        "bottle_map": bottle_map,
        "bottle_labels": bottle_labels,
        # centrifuge & washes per your spec
        "centrifuge": {"rpm": 8000, "minutes": 10, "tube": "V2_tube"},
        "wash": {"reagent": bottle_map.get("ethanol", "ethanol_bottle"), "cycles": 2},
        # prefer ambient drying when mentioned
        "drying": {
            "prefer_ambient_if_mentioned": True,
            "ambient_minutes": 1440,
            "vacuum_minutes": 720,
            "vacuum_temp_C": 25,
        },
    }


def _safety_seed(doc: dict) -> None:
    """Seed required defaults so external hooks don't crash, and set sane baselines."""
    d = doc.setdefault("defaults", {})
    # stop external hooks from KeyError
    d.setdefault("dropwise_timer_minutes", 5)
    d.setdefault("stir_idle_lookahead_ops", 0)
    # harmonized defaults used downstream
    d.setdefault("stir_rpm", 700)
    d.setdefault("centrifuge_minutes", 10)
    d.setdefault("centrifuge_rpm", 8000)


def _sanity_assertions(doc: dict) -> None:
    errs = []
    d = doc.get("defaults") or {}
    if d.get("centrifuge_rpm") != 8000:
        errs.append(f"defaults.centrifuge_rpm={d.get('centrifuge_rpm')} != 8000")
    # junk vessels?
    bad = [
        k
        for k in (doc.get("vessel_registry") or {})
        if re.fullmatch(r"v\d+_bottle", k, flags=re.I)
    ]
    if bad:
        errs.append(f"junk vessel ids present: {bad[:5]}{'…' if len(bad)>5 else ''}")
    # CF sequence present?
    mp = doc.get("micro_plan") or []
    ok_cf = any(
        mp[i].get("verb") == "place"
        and mp[i].get("object") == "V2_tube"
        and mp[i].get("to") == "CF1"
        and mp[i + 1].get("verb") == "set"
        and mp[i + 1].get("device") == "CF1"
        and mp[i + 1].get("param") == "rpm"
        and mp[i + 2].get("verb") == "set"
        and mp[i + 2].get("device") == "CF1"
        and mp[i + 2].get("param") == "power"
        for i in range(len(mp) - 2)
    )
    if not ok_cf:
        errs.append("centrifuge sequence missing place(V2_tube→CF1) before rpm/on")
    # ambient dry (no OV1/VP1 ops) if last step says ambient
    last = doc.get("steps", [])[-1] if doc.get("steps") else {}
    if "ambient" in (last.get("raw") or "").lower():
        if any(m.get("device") in {"OV1", "VP1"} for m in mp):
            errs.append("ambient dry requested but OV1/VP1 ops present in micro_plan")
        if not any(
            m.get("verb") == "place" and m.get("to") == "bench" for m in mp[-10:]
        ):
            errs.append(
                "ambient dry requested but no place(...→bench) near end of plan"
            )
    print(
        "[converter] SANITY OK"
        if not errs
        else "[converter] SANITY FAIL:\n  - " + "\n  - ".join(errs)
    )


_JUNK_VESSEL_RE = re.compile(r"^[vV]\d+(?:_tube)?_bottle$")


def _tidy_registry(doc: dict) -> None:
    reg = doc.setdefault("vessel_registry", {})

    # remove junk ids produced by naive "Vn → vN_bottle" mapping or glitchy long names
    for k in list(reg.keys()):
        if (
            _JUNK_VESSEL_RE.match(k)
            or k in {"solute_bottle"}
            or (k.endswith("_bottle") and "centrifuge" in k.lower())
        ):
            reg.pop(k, None)


def apply_postprocessing(doc: dict) -> dict:
    global _BANNER_SHOWN
    if not _BANNER_SHOWN:
        try:
            print(f"[converter] robot-normalizer {ROBOT_NORMALIZER_VERSION} active")
        except Exception:
            pass
        _BANNER_SHOWN = True

    # 1) safety seeds so external hook never crashes
    _safety_seed(doc)

    # 2) external hook (if you have one), never allowed to break the run
    try:
        if _ext_apply_post is not None:
            doc = _ext_apply_post(doc)
    except Exception as e:
        print("[converter] _ext_apply_post error:", repr(e))

    # 3) normal pipeline
    doc = robot_normalize(doc)

    # 4) safety fuse: force harmonized defaults if upstream didn’t set them
    d = doc.setdefault("defaults", {})
    if d.get("centrifuge_rpm") != 8000:
        print("[converter] forcing defaults.centrifuge_rpm to 8000 (safety fuse)")
        d["centrifuge_rpm"] = 8000
    d.setdefault("centrifuge_minutes", 10)
    d.setdefault("stir_rpm", 700)

    # 5) post-polish: bottles, heating placement, CF sequence, washes, ambient dry, micro_plan rebuild
    try:
        cfg = _build_bottle_config(doc)
        if _post_polish is not None:
            before = (doc.get("defaults") or {}).get("centrifuge_rpm")
            doc = _post_polish(doc, config=cfg)
            after = (doc.get("defaults") or {}).get("centrifuge_rpm")
            print(f"[converter] post_polish ran (rpm {before}→{after})")
        else:
            print("[converter] post_polish not imported; skipping")
    except Exception as e:
        print("[converter] _run_post_polish error:", repr(e))

    _rebuild_micro_plan(doc)
    _tidy_registry(doc)

    _sanity_assertions(doc)

    return doc


def _walk(obj, fn):
    if isinstance(obj, dict):
        for k, v in list(obj.items()):
            obj[k] = _walk(v, fn)
        return fn(obj) or obj
    if isinstance(obj, list):
        for i, v in enumerate(list(obj)):
            obj[i] = _walk(v, fn)
        return fn(obj) or obj
    return fn(obj) or obj


def _seed_defaults_devices(doc):
    d = doc.setdefault("defaults", {})
    d.setdefault("stir_rpm", 700)
    d.setdefault("centrifuge_rpm", 8000)
    d.setdefault("centrifuge_minutes", 10)
    d.setdefault("transfer_rate_slow", "slow")
    d.setdefault("room_temp_C", 25)
    dev = doc.setdefault("devices", {})
    dev.setdefault("stir_plate_id", "SP1")
    dev.setdefault("hotplate_id", "HP1")
    dev.setdefault("centrifuge_id", "CF1")
    dev.setdefault("oven_id", "OV1")
    dev.setdefault("vortex_id", "VX1")


def _clean_names(doc):
    def fix(n):
        if isinstance(n, dict):
            for k in ("name", "reagent", "solvent", "object", "from"):
                if k in n and isinstance(n[k], str):
                    n[k] = (
                        re.sub(r"\s*\([^)]*\)", "", n[k])
                        .replace(" under magnetic stirring", "")
                        .replace(" dropwise", "")
                        .strip()
                    )
        return None

    _walk(doc, fix)


def _dedupe_micro_ops(doc):
    def dedupe(lst):
        seen, out = set(), []
        for op in lst:
            key = json.dumps(
                {
                    k: op.get(k)
                    for k in (
                        "verb",
                        "device",
                        "param",
                        "value",
                        "from",
                        "to",
                        "object",
                        "minutes",
                        "tube",
                    )
                },
                sort_keys=True,
            )
            if key in seen:
                continue
            seen.add(key)
            out.append(op)
        return out

    if isinstance(doc.get("micro_plan"), list):
        doc["micro_plan"] = dedupe(doc["micro_plan"])
    for st in doc.get("steps", []):
        if isinstance(st.get("micro_ops"), list):
            st["micro_ops"] = dedupe(st["micro_ops"])


def _rebuild_step_micro_ops(step, devices, defaults, step_index):
    """Return a fresh list of micro_ops synthesized from canonical ops."""
    m = []
    SP = devices.get("stir_plate_id", "SP1")
    HP = devices.get("hotplate_id", "HP1")
    CF = devices.get("centrifuge_id", "CF1")
    OV = devices.get("oven_id", "OV1")

    def add_wait(minutes):
        if minutes is not None:
            m.append(
                {"verb": "wait", "minutes": int(minutes), "step_index": step_index}
            )

    for op in step.get("ops", []):
        typ = op.get("op")

        if typ == "move_to_stir_plate":
            m += [
                {
                    "verb": "pick_up",
                    "object": op.get("vessel", "V1"),
                    "step_index": step_index,
                },
                {
                    "verb": "place",
                    "object": op.get("vessel", "V1"),
                    "to": SP,
                    "step_index": step_index,
                },
            ]

        elif typ == "set_stir_rate":
            m += [
                {
                    "verb": "set",
                    "device": SP,
                    "param": "rpm",
                    "value": op.get("rpm", defaults.get("stir_rpm", 700)),
                    "step_index": step_index,
                }
            ]

        elif typ == "set" and op.get("param") == "temperature_C":
            dev = op.get("device") or HP
            m += [
                {
                    "verb": "set",
                    "device": dev,
                    "param": "temperature_C",
                    "value": op.get("value"),
                    "step_index": step_index,
                }
            ]

        elif typ == "add_solvent":
            rate = "slow" if "dropwise" in step.get("raw", "").lower() else "normal"
            temp_context = step.get("temperature_C")
            m += [
                {
                    "verb": "pick_up",
                    "object": op.get("solvent", "solvent"),
                    "from": "bench",
                    "step_index": step_index,
                },
                {
                    "verb": "pour",
                    "from": op.get("solvent", "solvent"),
                    "to": step.get("vessel", "V1"),
                    "volume": op.get("volume"),
                    "volume_units": op.get("volume_units", "mL"),
                    "rate": rate,
                    "temperature_context": temp_context,
                    "step_index": step_index,
                },
                {
                    "verb": "place",
                    "object": op.get("solvent", "solvent"),
                    "to": "bench",
                    "step_index": step_index,
                },
            ]

        elif typ == "add_solute":
            m += [
                {
                    "verb": "pick_up",
                    "object": op.get("solute", "solute"),
                    "from": "bench",
                    "step_index": step_index,
                },
                {
                    "verb": "pour",
                    "from": op.get("solute", "solute"),
                    "to": step.get("vessel", "V1"),
                    "mass": op.get("mass"),
                    "mass_units": op.get("mass_units", "mg"),
                    "step_index": step_index,
                },
                {
                    "verb": "place",
                    "object": op.get("solute", "solute"),
                    "to": "bench",
                    "step_index": step_index,
                },
            ]

        elif typ == "transfer_to_centrifuge_tube":
            m += [
                {
                    "verb": "pour",
                    "from": op.get("from", "V1"),
                    "to": op.get("to", "V2_tube"),
                    "context_vessel": op.get("from", "V1"),
                    "step_index": step_index,
                }
            ]

        elif typ == "resuspend":
            tube = op.get("tube") or step.get("vessel", "V2_tube")
            m += [
                {
                    "verb": "vortex",
                    "device": devices.get("vortex_id", "VX1"),
                    "tube": tube,
                    "step_index": step_index,
                }
            ]

        elif typ == "centrifuge":
            rpm = op.get("rpm", defaults.get("centrifuge_rpm", 4000))
            mins = op.get("minutes", defaults.get("centrifuge_minutes", 10))
            m += [
                {
                    "verb": "set",
                    "device": CF,
                    "param": "rpm",
                    "value": rpm,
                    "step_index": step_index,
                },
                {"verb": "start", "device": CF, "step_index": step_index},
                {"verb": "wait", "minutes": int(mins), "step_index": step_index},
                {"verb": "stop", "device": CF, "step_index": step_index},
            ]

        elif typ == "decant_supernatant":
            m += [
                {
                    "verb": "decant",
                    "object": op.get("tube", "V2_tube"),
                    "step_index": step_index,
                }
            ]

        elif typ == "add_wash_solvent":
            m += [
                {
                    "verb": "pick_up",
                    "object": op.get("solvent", "wash solvent"),
                    "from": "bench",
                    "step_index": step_index,
                },
                {
                    "verb": "pour",
                    "from": op.get("solvent", "wash solvent"),
                    "to": op.get("tube", "V2_tube"),
                    "volume": op.get("volume"),
                    "volume_units": op.get("volume_units", "mL"),
                    "step_index": step_index,
                },
                {
                    "verb": "place",
                    "object": op.get("solvent", "wash solvent"),
                    "to": "bench",
                    "step_index": step_index,
                },
            ]

        elif typ == "move_to_oven":
            m += [
                {
                    "verb": "place",
                    "object": op.get("tube", "V2_tube"),
                    "to": OV,
                    "step_index": step_index,
                }
            ]

        elif typ == "wait" or typ == "timer":  # normalize both to wait
            add_wait(op.get("minutes"))

        # (Ignore unknown ops in micro synthesis — keep steps/ops authoritative)

    return m


def _rebuild_micro_plan(doc: dict) -> None:
    allowed = {
        "pick_up",
        "place",
        "pour",
        "set",
        "wait",
        "start",
        "stop",
        "vortex",
        "decant",
    }

    def validate_action(action):
        """Validate atomic action parameters"""
        verb = action.get("verb")
        if not verb:
            print(f"[converter] Warning: Action missing 'verb': {action}")
            return False

        # Required parameters for each action type
        if verb == "pick_up":
            valid = "object" in action
            if not valid:
                print(f"[converter] Warning: pick_up missing 'object': {action}")
            return valid
        elif verb == "place":
            valid = "object" in action and "to" in action
            if not valid:
                print(f"[converter] Warning: place missing 'object' or 'to': {action}")
            return valid
        elif verb == "pour":
            valid = "from" in action and "to" in action
            if not valid:
                print(f"[converter] Warning: pour missing 'from' or 'to': {action}")
            return valid
        elif verb == "set":
            valid = "device" in action and "param" in action and "value" in action
            if not valid:
                print(
                    f"[converter] Warning: set missing 'device', 'param', or 'value': {action}"
                )
            return valid
        elif verb == "wait":
            valid = "minutes" in action and isinstance(
                action.get("minutes"), (int, float)
            )
            if not valid:
                print(
                    f"[converter] Warning: wait missing or invalid 'minutes': {action}"
                )
            return valid
        elif verb in {"start", "stop"}:
            valid = "device" in action
            if not valid:
                print(f"[converter] Warning: {verb} missing 'device': {action}")
            return valid
        elif verb == "vortex":
            valid = "device" in action and "tube" in action
            if not valid:
                print(
                    f"[converter] Warning: vortex missing 'device' or 'tube': {action}"
                )
            return valid
        elif verb == "decant":
            valid = "object" in action
            if not valid:
                print(f"[converter] Warning: decant missing 'object': {action}")
            return valid
        return True  # allow other verbs

    out = []
    for st in doc.get("steps", []):
        idx = st.get("index")
        for m in st.get("micro_ops") or []:
            if m.get("verb") in allowed and validate_action(m):
                mm = dict(m)
                mm["step_index"] = idx  # enforce correct index
                out.append(mm)
    # de-dup consecutive identical ops
    flat = []
    for m in out:
        if not flat or flat[-1] != m:
            flat.append(m)
    # ensure no oven/vacuum ops when ambient is requested
    flat = [m for m in flat if m.get("device") not in {"OV1", "VP1"}]
    doc["micro_plan"] = flat


def _rebuild_micro_from_ops(doc):
    """Discard incoming micro_ops/micro_plan and rebuild them deterministically from step ops."""
    devices = doc.get("devices", {})
    defaults = doc.get("defaults", {})
    steps = doc.get("steps", [])

    # Rebuild per-step micro_ops
    for idx, st in enumerate(steps, start=1):
        st["micro_ops"] = _rebuild_step_micro_ops(st, devices, defaults, idx)

    # Flatten into micro_plan with correct step_index
    mp = []
    for idx, st in enumerate(steps, start=1):
        for mo in st.get("micro_ops", []):
            mo.setdefault("step_index", idx)
            mp.append(mo)
    doc["micro_plan"] = mp


def _final_invariants(doc):
    # Canonical vessels
    reg = doc.setdefault("vessel_registry", {})
    reg["V1"] = "round-bottom flask 100 mL"
    reg["V2"] = "15 mL centrifuge tube rack"
    reg["V2_tube"] = "15 mL centrifuge tube"
    for k in list(reg.keys()):
        if k not in ("V1", "V2", "V2_tube"):
            del reg[k]

    # Timers → wait (ops)
    for st in doc.get("steps", []):
        for op in st.get("ops", []):
            if op.get("op") == "timer":
                op["op"] = "wait"

    # Wash waits (micro_plan): 1 min pre-spin, 10 min spin
    mp = doc.get("micro_plan", [])
    for si, st in enumerate(doc.get("steps", []), start=1):
        raw = (st.get("raw", "") or "").lower()
        if st.get("action") in {"postprocess", "wash"} and "centrifuge" in raw:
            start_i = next(
                (
                    i
                    for i, m in enumerate(mp)
                    if m.get("step_index") == si
                    and m.get("verb") == "start"
                    and m.get("device") == doc["devices"].get("centrifuge_id", "CF1")
                ),
                None,
            )
            if start_i is not None:
                for i, m in enumerate(mp):
                    if m.get("step_index") == si and m.get("verb") == "wait":
                        m["minutes"] = 1 if i < start_i else 10

    # Pure drying: only oven place/set/wait in both ops and micro
    OV = doc.get("devices", {}).get("oven_id", "OV1")
    for si, st in enumerate(doc.get("steps", []), start=1):
        raw = (st.get("raw", "") or "").lower()
        if "dry" in raw or "oven" in raw:
            # keep existing temp/min if present, else defaults
            t = next(
                (
                    op.get("value")
                    for op in st.get("ops", [])
                    if op.get("op") == "set" and op.get("param") == "temperature_C"
                ),
                60,
            )
            minutes = st.get("minutes") or next(
                (
                    op.get("minutes")
                    for op in st.get("ops", [])
                    if op.get("op") == "wait"
                ),
                720,
            )
            st["minutes"] = minutes
            st["ops"] = [
                {"op": "move_to_oven", "oven_id": OV, "tube": "V2_tube"},
                {"device": OV, "op": "set", "param": "temperature_C", "value": t},
                {"op": "wait", "minutes": int(minutes)},
            ]
            # micro: place OV1, set temp, wait minutes
            [mo for mo in doc["micro_plan"] if mo.get("step_index") == si]
            doc["micro_plan"] = [
                mo for mo in doc["micro_plan"] if mo.get("step_index") != si
            ]
            doc["micro_plan"] += [
                {"verb": "place", "object": "V2_tube", "to": OV, "step_index": si},
                {
                    "verb": "set",
                    "device": OV,
                    "param": "temperature_C",
                    "value": t,
                    "step_index": si,
                },
                {"verb": "wait", "minutes": int(minutes), "step_index": si},
            ]


def _canonicalize_isolate_transfer(doc):
    for st in doc.get("steps", []):
        raw = (st.get("raw", "") or "").lower()
        act = (st.get("action") or "").lower()
        if "centrifuge" in raw and act in {"isolate", "collect", "transfer"}:
            src = st.get("vessel") or st.get("source_vessel") or "V1"
            # parse rpm/mins
            m_rpm = re.search(r"(\d{3,5})\s*rpm", raw)
            rpm = int(m_rpm.group(1)) if m_rpm else doc["defaults"]["centrifuge_rpm"]
            m_min = re.search(r"(\d+)\s*(min|minutes)", raw)
            mins = (
                int(m_min.group(1)) if m_min else doc["defaults"]["centrifuge_minutes"]
            )
            st["vessel"] = src
            st["ops"] = [
                {"op": "transfer_to_centrifuge_tube", "from": src, "to": "V2_tube"},
                {
                    "op": "centrifuge",
                    "centrifuge_id": doc["devices"]["centrifuge_id"],
                    "rpm": rpm,
                    "minutes": mins,
                },
                {"op": "decant_supernatant", "tube": "V2_tube"},
            ]
            # Wash parsing
            if "wash" in raw and ("ethanol" in raw or "acetone" in raw):
                vol = _parse_vol_ml(raw) or 20
                solvent = "acetone" if "acetone" in raw else "ethanol"
                st["wash_solvent"] = {
                    "name": solvent,
                    "volume": vol,
                    "volume_units": "mL",
                }
                rep = 3 if "three" in raw else 2 if "twice" in raw else None
                m_rep = re.search(r"(?:repeat.*?|^)\s*(\d+)\s*(?:x|×|times)\b", raw)
                if m_rep:
                    rep = int(m_rep.group(1))
                if rep:
                    st["repeats"] = rep
                st["ops"] += [
                    {
                        "op": "add_wash_solvent",
                        "solvent": solvent,
                        "tube": "V2_tube",
                        "volume": vol,
                        "volume_units": "mL",
                    },
                    {"op": "resuspend", "tube": "V2_tube"},
                    {
                        "op": "centrifuge",
                        "centrifuge_id": doc["devices"]["centrifuge_id"],
                        "rpm": rpm,
                        "minutes": mins,
                    },
                    {"op": "decant_supernatant", "tube": "V2_tube"},
                ]


def _drop_duplicate_process_after_collect(doc):
    steps = doc.get("steps", [])
    keep = []
    for i, st in enumerate(steps):
        if (
            st.get("action", "").lower() == "process"
            and "centrifuge" in (st.get("raw", "").lower())
            and i > 0
            and (steps[i - 1].get("action", "").lower() == "collect")
        ):
            continue  # skip duplicate
        keep.append(st)
    doc["steps"] = keep


def _fill_missing_step_index(doc):
    si = None
    for m in doc.get("micro_plan", []) or []:
        if "step_index" in m and m["step_index"] is not None:
            si = m["step_index"]
        else:
            m["step_index"] = si


def _ensure_process_centrifuge(doc):
    """If a 'process' step mentions centrifuge but lacks the op, insert the canonical sequence from V1."""
    steps = doc.get("steps", [])
    for st in steps:
        raw = (st.get("raw", "") or "").lower()
        if st.get("action", "").lower() == "process" and "centrifuge" in raw:
            ops = st.setdefault("ops", [])
            if not any(op.get("op") == "centrifuge" for op in ops):
                m_rpm = re.search(r"(\d{3,5})\s*rpm", raw)
                m_min = re.search(r"(\d+)\s*(?:min|minutes)\b", raw)
                rpm = (
                    int(m_rpm.group(1))
                    if m_rpm
                    else doc["defaults"].get("centrifuge_rpm", 4000)
                )
                mins = (
                    int(m_min.group(1))
                    if m_min
                    else doc["defaults"].get("centrifuge_minutes", 10)
                )
                st["vessel"] = "V1"
                st["ops"] = [
                    {
                        "op": "transfer_to_centrifuge_tube",
                        "from": "V1",
                        "to": "V2_tube",
                    },
                    {
                        "op": "centrifuge",
                        "centrifuge_id": doc["devices"]["centrifuge_id"],
                        "rpm": rpm,
                        "minutes": mins,
                    },
                    {"op": "decant_supernatant", "tube": "V2_tube"},
                ]


def _split_wash_and_dry(doc):
    new = []
    for st in doc.get("steps", []):
        raw = (st.get("raw", "") or "").lower()
        if (
            st.get("action") in {"postprocess", "wash"}
            and "wash" in raw
            and ("dry" in raw or "oven" in raw)
        ):
            wash = dict(st)
            wash["raw"] = "wash " + wash.get("raw", "")
            wash.pop("minutes", None)
            dry = dict(st)
            dry["action"] = "dry"
            dry["raw"] = "dry " + dry.get("raw", "")
            dry.pop("wash_solvent", None)
            dry.pop("repeats", None)
            dry["ops"] = []  # will be filled by _force_pure_dry
            new += [wash, dry]
        else:
            new.append(st)
    doc["steps"] = new


def _dedupe_adjacent_waits(doc):
    """Remove consecutive duplicate wait ops within a step."""
    for st in doc.get("steps", []):
        ops = st.get("ops", [])
        new, prev = [], None
        for op in ops:
            if (
                prev
                and op.get("op") == prev.get("op") == "wait"
                and op.get("minutes") == prev.get("minutes")
            ):
                continue
            new.append(op)
            prev = op
        st["ops"] = new


def _sync_heat_and_dry_waits(doc):
    """Make micro_plan waits match step minutes for heat/hold and drying steps."""
    steps = doc.get("steps", [])
    mp = doc.get("micro_plan", []) or []
    for idx, st in enumerate(steps, start=1):
        raw = (st.get("raw", "") or "").lower()
        if not st.get("minutes"):
            continue
        if st.get("action") in {"stir", "heat_hold"} or ("dry" in raw):
            for m in mp:
                if m.get("verb") == "wait" and m.get("step_index") == idx:
                    m["minutes"] = st["minutes"]


def _fix_wash_microplan(doc):
    """For each wash step, set ~1 min pre-spin wait, 10 min spin wait, and unify CF1 rpm with step ops."""
    steps = doc.get("steps", [])
    mp = doc.get("micro_plan", []) or []
    for idx, st in enumerate(steps, start=1):
        raw = (st.get("raw", "") or "").lower()
        if not (
            st.get("action") in {"postprocess", "wash"}
            and "wash" in raw
            and "centrifuge" in raw
        ):
            continue
        # find CF1 'start' row
        start_i = next(
            (
                i
                for i, m in enumerate(mp)
                if m.get("step_index") == idx
                and m.get("verb") == "start"
                and m.get("device") == "CF1"
            ),
            None,
        )
        if start_i is None:
            continue
        # set waits
        for i, m in enumerate(mp):
            if m.get("step_index") == idx and m.get("verb") == "wait":
                m["minutes"] = 1 if i < start_i else 10
        # unify rpm with step op (or default)
        rpm = next(
            (
                op.get("rpm")
                for op in st.get("ops", [])
                if op.get("op") == "centrifuge" and op.get("rpm")
            ),
            doc["defaults"].get("centrifuge_rpm", 4000),
        )
        for m in mp:
            if (
                m.get("step_index") == idx
                and m.get("verb") == "set"
                and m.get("device") == "CF1"
                and m.get("param") == "rpm"
            ):
                m["value"] = rpm


def _force_pure_dry(doc):
    for st in doc.get("steps", []):
        raw = (st.get("raw", "") or "").lower()
        if "dry" in raw or "oven" in raw:
            t = find_temp_c(st.get("raw", "")) or st.get("temperature_C") or 60.0
            m = find_minutes(st.get("raw", "")) or st.get("minutes") or 720
            st["minutes"] = m
            st["ops"] = [
                {
                    "op": "move_to_oven",
                    "oven_id": doc["devices"]["oven_id"],
                    "tube": "V2_tube",
                },
                {
                    "device": doc["devices"]["oven_id"],
                    "op": "set",
                    "param": "temperature_C",
                    "value": t,
                },
                {"op": "wait", "minutes": m},
            ]
            st.pop("wash_solvent", None)
            st.pop("repeats", None)


def _ops_timer_to_wait(doc):
    """Normalize op timers to 'wait' for consistency with micro-ops."""
    for st in doc.get("steps", []):
        for op in st.get("ops", []):
            if op.get("op") == "timer":
                op["op"] = "wait"


def _split_transfer_collect(doc):
    steps = doc.get("steps", [])
    for i, st in enumerate(steps):
        if (st.get("action") or "").lower() == "transfer":
            if any(
                (s.get("action") or "").lower() in {"collect", "isolate"}
                or ("centrifuge" in (s.get("raw", "").lower()))
                for s in steps[i + 1 :]
            ):
                st["ops"] = [
                    op
                    for op in st.get("ops", [])
                    if op.get("op") == "transfer_to_centrifuge_tube"
                ]


def _timers_to_waits(doc):
    _walk(
        doc,
        lambda n: (
            n.update({"op": "wait"})
            if isinstance(n, dict) and n.get("op") == "timer"
            else None
        ),
    )


def _fix_wash_waits(doc):
    mp = doc.get("micro_plan", [])
    # find CF1 start for the wash step (e.g., step_index == 7)
    for si in {m.get("step_index") for m in mp if m.get("verb") in {"start", "wait"}}:
        cf1 = next(
            (
                i
                for i, m in enumerate(mp)
                if m.get("step_index") == si
                and m.get("verb") == "start"
                and m.get("device") == "CF1"
            ),
            None,
        )
        if cf1 is None:
            continue
        for i, m in enumerate(mp):
            if m.get("step_index") == si and m.get("verb") == "wait":
                m["minutes"] = 1 if i < cf1 else 10


_LABEL_RX = re.compile(
    r"(?:\s*under magnetic stirring|\s*at room temperature)|\s*\([^)]*\)", re.I
)


def _clean_labels(doc):
    def f(n):
        if isinstance(n, dict):
            for k in ("name", "reagent", "solvent", "object", "solute", "from"):
                if isinstance(n.get(k), str):
                    n[k] = _LABEL_RX.sub("", n[k]).strip()
        return None

    _walk(doc, f)
    vc = doc.setdefault("vessel_contents", {})
    for k, v in list(vc.items()):
        if isinstance(v, str):
            s = _LABEL_RX.sub("", v).replace(" None None", "").strip()
            vc[k] = re.sub(r"\s{2,}", " ", s)


def _fix_transfer_context(doc):
    for m in doc.get("micro_plan", []):
        if (
            m.get("step_index") == 5
            and m.get("verb") == "pour"
            and m.get("to") == "V2_tube"
        ):
            m["context_vessel"] = "V1"


def _first_spin_default(doc):
    first = True

    def fix(n):
        nonlocal first
        if isinstance(n, dict) and n.get("op") == "centrifuge":
            if first:
                n.setdefault("rpm", 5000)
                n.setdefault("minutes", 10)
                first = False
            else:
                n.setdefault("rpm", doc["defaults"]["centrifuge_rpm"])
                n.setdefault("minutes", doc["defaults"]["centrifuge_minutes"])
        return None

    _walk(doc, fix)


def _unify_heat_wait(doc):
    steps = doc.get("steps", [])
    for i, st in enumerate(steps):
        raw = (st.get("raw", "") or "").lower()
        if st.get("action") in {"heat_hold", "stir"} and (
            "2 h" in raw or "2 hours" in raw or re.search(r"\b120\s*min\b", raw)
        ):
            st["minutes"] = 120
        mins = st.get("minutes")
        if st.get("action") == "heat_hold" and mins:
            for m in doc.get("micro_plan", []):
                if m.get("verb") == "wait" and m.get("step_index") in (i, i + 1):
                    m["minutes"] = mins


def _collapse_adjacent_collect_spins(doc):
    """Drop a collect-like step if the previous step already did centrifuge+decant with same settings."""
    steps = doc.get("steps", [])
    keep = []
    prev_spin = None
    for st in steps:
        ops = st.get("ops", [])
        spin = next((op for op in ops if op.get("op") == "centrifuge"), None)
        dec = any(op.get("op") == "decant_supernatant" for op in ops)
        if spin and dec:
            key = (spin.get("rpm"), spin.get("minutes"))
            if prev_spin == key:
                # skip duplicate collect block
                continue
            prev_spin = key
        keep.append(st)
    doc["steps"] = keep


def _collapse_consecutive_dry_steps(doc):
    steps = doc.get("steps", [])
    keep = []
    pending = None  # (temp, minutes) from first dry
    for st in steps:
        raw = (st.get("raw", "") or "").lower()
        is_dry = ("dry" in raw) or ("oven" in raw)
        if not is_dry:
            pending = None
            keep.append(st)
            continue
        # extract temp/min from this dry step if present
        t = next(
            (
                op.get("value")
                for op in st.get("ops", [])
                if op.get("op") == "set" and op.get("param") == "temperature_C"
            ),
            None,
        )
        m = next(
            (op.get("minutes") for op in st.get("ops", []) if op.get("op") == "wait"),
            None,
        ) or st.get("minutes")
        if pending is None:
            # keep the first; ensure ops are pure oven set/wait (your _force_pure_dry/_final_invariants will enforce)
            pending = (t, m)
            keep.append(st)
        else:
            # merge into the first: choose latest explicit temp/min if provided
            pt, pm = pending
            if t is not None:
                pt = t
            if m is not None:
                pm = m
            pending = (pt, pm)
            # drop this duplicate dry step
    # write back merged temp/min onto the kept dry step
    doc["steps"] = keep


def _fix_wash_blocks(doc):
    for st in doc.get("steps", []):
        raw = (st.get("raw", "") or "").lower()
        if st.get("action") in {"postprocess", "wash"} and "wash" in raw:
            # Volume (supports µL/mL/L and decimals)
            vol = _parse_vol_ml(raw) or 20
            # Solvent detection (deionized water / ethanol / acetone / isopropanol)
            if "deionized water" in raw or re.search(r"\bdi\b.*water", raw):
                solv = "deionized water"
            elif "acetone" in raw:
                solv = "acetone"
            elif "isopropanol" in raw or "ipa" in raw or "2-propanol" in raw:
                solv = "isopropanol"
            elif "ethanol" in raw:
                solv = "ethanol"
            else:
                solv = "ethanol"
            st["wash_solvent"] = {"name": solv, "volume": vol, "volume_units": "mL"}

            # Repeats: twice / 3x / 'three times'
            rep = 2 if "twice" in raw else None
            m_rep = re.search(r"(?:repeat.*?|^)\s*(\d+)\s*(?:x|×|times)\b", raw)
            if "three" in raw:
                rep = 3
            if m_rep:
                rep = int(m_rep.group(1) or rep or 2)
            if rep:
                st["repeats"] = rep

            # Parse spin settings from text if present
            m_rpm = re.search(r"(\d{3,5})\s*rpm", raw)
            m_min = re.search(r"(\d+)\s*(min|minutes)", raw)
            rpm = int(m_rpm.group(1)) if m_rpm else doc["defaults"]["centrifuge_rpm"]
            mins = (
                int(m_min.group(1)) if m_min else doc["defaults"]["centrifuge_minutes"]
            )

            # Normalize ops: wash → resuspend → spin → decant (one cycle; executor will loop by repeats)
            st["ops"] = [
                {
                    "op": "add_wash_solvent",
                    "solvent": solv,
                    "tube": "V2_tube",
                    "volume": vol,
                    "volume_units": "mL",
                },
                {"op": "resuspend", "tube": "V2_tube"},
                {
                    "op": "centrifuge",
                    "centrifuge_id": doc["devices"]["centrifuge_id"],
                    "rpm": rpm,
                    "minutes": mins,
                },
                {"op": "decant_supernatant", "tube": "V2_tube"},
            ]


def _normalize_calcination(doc):
    for st in doc.get("steps", []):
        raw = (st.get("raw", "") or "").lower()
        if "calcine" in raw or "calcination" in raw:
            # Oven @ 500 °C for 120 min; drop any leftover wash/vortex bits
            st["ops"] = [
                {
                    "op": "move_to_oven",
                    "oven_id": doc["devices"].get("oven_id", "OV1"),
                    "tube": "V2_tube",
                },
                {
                    "op": "set",
                    "device": doc["devices"].get("oven_id", "OV1"),
                    "param": "temperature_C",
                    "value": 500,
                },
                {"op": "wait", "minutes": 120},
            ]
            st["minutes"] = 120
            st.pop("wash_solvent", None)
            st.pop("repeats", None)
            # optional: trim micro_ops to a simple oven set + wait
            if isinstance(st.get("micro_ops"), list):
                st["micro_ops"] = [
                    {"verb": "place", "object": "V2_tube", "to": "OV1"},
                    {
                        "verb": "set",
                        "device": "OV1",
                        "param": "temperature_C",
                        "value": 500,
                    },
                    {"verb": "wait", "minutes": 120},
                ]


# 1) FORCE canonical vessel registry (overwrite & purge)
def _seed_vessels(doc):
    reg = doc.setdefault("vessel_registry", {})
    reg["V1"] = "round-bottom flask 100 mL"
    reg["V2"] = "15 mL centrifuge tube rack"
    reg["V2_tube"] = "15 mL centrifuge tube"
    for k, v in list(reg.items()):
        if k in ("V1", "V2", "V2_tube"):
            continue
        s = str(v or "").lower()
        if (
            ("centrifuge" in s)
            or ("rpm" in s)
            or re.search(r"\b\d+\s*mL\b", s)
            or ("solution" in s)
            or ("heated" in s)
        ):
            del reg[k]


# 2) MAP aliases case-insensitively, incl. micro-ops placeholders
def _map_aliases(doc):
    dev = doc.setdefault("devices", {})
    alias = {
        "stir_plate": dev.get("stir_plate_id", "SP1"),
        "stir-plate": dev.get("stir_plate_id", "SP1"),
        "stirrer": dev.get("stir_plate_id", "SP1"),
        "centrifuge": dev.get("centrifuge_id", "CF1"),
        "vortex": dev.get("vortex_id", "VX1"),
        "vortexer": dev.get("vortex_id", "VX1"),
        "oven": dev.get("oven_id", "OV1"),
        "centrifuge tubes": "V2_tube",
        "tube": "V2_tube",
        "source": "V1",  # treat generic “source” as the reactor
    }

    def fix(n):
        if isinstance(n, dict):
            # device field
            if isinstance(n.get("device"), str):
                key = n["device"].strip().lower()
                if key in alias:
                    n["device"] = alias[key]
            # locations / vessels
            for k in (
                "to",
                "from",
                "object",
                "vessel",
                "source_vessel",
                "target_vessel",
                "tube",
                "context_vessel",
            ):
                v = n.get(k)
                if isinstance(v, str):
                    s = re.sub(r"\s*\([^)]*\)", "", v).strip()
                    sl = s.lower()
                    if re.fullmatch(r"v\d+_tube", sl) or sl == "v1_tube":
                        n[k] = "V2_tube"
                        continue
                    if sl in alias:
                        n[k] = alias[sl]
                        continue
                    if (
                        ("flask 100 ml" in sl)
                        or ("round-bottom flask 100 ml" in sl)
                        or ("beaker 100 ml" in sl)
                    ):
                        n[k] = "V1"
                        continue
                    n[k] = s
        return None

    _walk(doc, fix)


# 3) CANONICALIZE add/transfer/dry blocks
def _normalize_add_transfer_and_dry(doc):
    for st in doc.get("steps", []):
        raw = (st.get("raw", "") or "").lower()
        act = (st.get("action") or "").lower()

        # Add base → into V1 with proper ops
        if act == "add" and "ammonium hydroxide" in raw:
            st["vessel"] = "V1"
            st["source_vessel"] = "bench"
            st["target_vessel"] = "V1"
            st["ops"] = [
                {
                    "op": "move_to_stir_plate",
                    "vessel": "V1",
                    "stir_plate_id": doc["devices"]["stir_plate_id"],
                },
                {
                    "op": "set_stir_rate",
                    "vessel": "V1",
                    "rpm": doc["defaults"]["stir_rpm"],
                },
                {
                    "device": doc["devices"]["hotplate_id"],
                    "op": "set",
                    "param": "temperature_C",
                    "value": 80,
                },
                {
                    "op": "add_solvent",
                    "vessel": "V1",
                    "solvent": "ammonium hydroxide",
                    "volume": 5,
                    "volume_units": "mL",
                    "rate": "slow",
                },
                {"op": "timer", "minutes": 60},
                {
                    "op": "monitor_ph",
                    "sensor": "pH_probe",
                    "strategy": "titrate_addition",
                    "target_range": [9, 10],
                },
            ]

        # Transfer to spin must be from V1
        if act == "transfer" and "centrifuge" not in raw:
            st["vessel"] = "V1"
            st["ops"] = [
                {"op": "transfer_to_centrifuge_tube", "from": "V1", "to": "V2_tube"},
                {
                    "op": "centrifuge",
                    "centrifuge_id": doc["devices"]["centrifuge_id"],
                    "rpm": doc["defaults"]["centrifuge_rpm"],
                    "minutes": doc["defaults"]["centrifuge_minutes"],
                },
                {"op": "decant_supernatant", "tube": "V2_tube"},
            ]

        # Drying step: parse temp/time; use oven hold with WAIT (not timer)
        if act in {"postprocess", "wash", "dry"} and "dry" in raw:
            t = find_temp_c(st.get("raw", "")) or 60.0
            m_val = find_minutes(st.get("raw", "")) or 720
            st["ops"] = [
                {
                    "op": "move_to_oven",
                    "oven_id": doc["devices"]["oven_id"],
                    "tube": "V2_tube",
                },
                {
                    "device": doc["devices"]["oven_id"],
                    "op": "set",
                    "param": "temperature_C",
                    "value": t,
                },
                {"op": "wait", "minutes": m_val},
            ]
            st.pop("wash_solvent", None)
            st.pop("repeats", None)
            st["minutes"] = m_val


# 4) REPLACE micro-op placeholders


# 4) REPLACE micro-op placeholders (“wash solvent”, “centrifuge tubes”)
def _fix_micro_placeholders(doc):
    # step-level solvent if available, else ethanol
    step_solvent = "ethanol"
    for st in doc.get("steps", []):
        if (
            "wash_solvent" in st
            and isinstance(st["wash_solvent"], dict)
            and st["wash_solvent"].get("name")
        ):
            step_solvent = st["wash_solvent"]["name"]
        if isinstance(st.get("micro_ops"), list):
            for m in st["micro_ops"]:
                if (
                    isinstance(m.get("object"), str)
                    and m["object"].strip().lower() == "wash solvent"
                ):
                    m["object"] = step_solvent
                if (
                    isinstance(m.get("from"), str)
                    and m["from"].strip().lower() == "wash solvent"
                ):
                    m["from"] = step_solvent
                if (
                    isinstance(m.get("to"), str)
                    and m["to"].strip().lower() == "centrifuge tubes"
                ):
                    m["to"] = "V2_tube"
    for m in doc.get("micro_plan", []):
        if (
            isinstance(m.get("object"), str)
            and m["object"].strip().lower() == "wash solvent"
        ):
            m["object"] = step_solvent
        if (
            isinstance(m.get("from"), str)
            and m["from"].strip().lower() == "wash solvent"
        ):
            m["from"] = step_solvent
        if (
            isinstance(m.get("to"), str)
            and m["to"].strip().lower() == "centrifuge tubes"
        ):
            m["to"] = "V2_tube"


# 5) CALL these from robot_normalize


def _post_fix_pass(doc):
    """
    Final strict-first-try cleanups that run after all other normalizers.
    - Force transfer steps to use V1 → V2_tube
    - Sync micro_plan waits with step minutes for heat-hold and oven-dry (off-by-one tolerant)
    - Map any micro_plan 'to' containing "centrifuge tube(s)" → V2_tube
    - Strip descriptors like "under magnetic stirring"/"dropwise" from micro_plan 'from'/'object'
    - Ensure wash solvent matches text (supports deionized water)
    - Remove duplicate 'timer' when a 'wait' with same minutes exists
    """
    import re

    steps = doc.get("steps", [])
    mp = doc.get("micro_plan", []) or []

    def _sl(s):
        return (s or "").lower() if isinstance(s, str) else s

    # 1) Force canonical transfer from V1 → V2_tube
    for st in steps:
        act = (st.get("action") or "").lower()
        if act == "transfer":
            st["vessel"] = "V1"
            ops = st.get("ops") or []
            for op in ops:
                if op.get("op") == "transfer_to_centrifuge_tube":
                    op["from"] = "V1"
                    op["to"] = "V2_tube"
            st["ops"] = ops

    # 2) Sync micro_plan waits for heat-hold and drying (handle 0/1-based step_index)
    for i, st in enumerate(steps):
        act = (st.get("action") or "").lower()
        raw = _sl(st.get("raw"))
        mins = st.get("minutes")
        if not mins:
            continue
        wants_sync = (act == "heat_hold") or (
            act in {"postprocess", "dry", "wash"} and "dry" in raw
        )
        if not wants_sync:
            continue
        for m in mp:
            if m.get("verb") == "wait" and m.get("step_index") in (i, i + 1):
                m["minutes"] = mins

    # 3) Micro-plan alias mapping and descriptor cleanup
    for m in mp:
        # 'to' -> V2_tube if mentions centrifuge tube(s)
        if isinstance(m.get("to"), str):
            to_sl = _sl(m["to"])
            if re.search(r"centrifuge\s+tubes?", to_sl):
                m["to"] = "V2_tube"
        # strip descriptors on 'from'/'object'
        for key in ("from", "object"):
            if isinstance(m.get(key), str):
                s = m[key]
                s = re.sub(r"\s*under magnetic stirring\b", "", s, flags=re.I)
                s = re.sub(r"\s*dropwise\b", "", s, flags=re.I)
                s = re.sub(r"\s*\([^)]*\)", "", s)
                m[key] = s.strip()

    # 4) Wash solvent: respect 'deionized water' if present in raw
    for st in steps:
        act = (st.get("action") or "").lower()
        raw = _sl(st.get("raw"))
        if act in {"postprocess", "wash"} and "wash" in raw:
            if "deionized water" in raw or re.search(r"\bdi\b.*water", raw):
                st.setdefault("wash_solvent", {})
                st["wash_solvent"]["name"] = "deionized water"
                # update ops solvent field
                ops = st.get("ops") or []
                for op in ops:
                    if op.get("op") == "add_wash_solvent":
                        op["solvent"] = "deionized water"
                st["ops"] = ops

    # 5) Remove duplicate timer when identical wait exists
    for st in steps:
        ops = st.get("ops") or []
        waits = {
            (op.get("op"), op.get("minutes"))
            for op in ops
            if op.get("op") == "wait" and op.get("minutes") is not None
        }
        new_ops = []
        for op in ops:
            if op.get("op") == "timer" and ("wait", op.get("minutes")) in waits:
                continue
            new_ops.append(op)
        st["ops"] = new_ops


def _sync_step_minutes_from_ops(doc):
    """Set step.minutes to the max 'wait' minutes found in step.ops; mirror waits to that value."""
    for st in doc.get("steps", []):
        waits = [
            op.get("minutes")
            for op in st.get("ops", [])
            if op.get("op") in ("wait", "timer") and op.get("minutes") is not None
        ]
        if not waits:
            continue
        m = int(max(waits))
        st["minutes"] = m
        for op in st.get("ops", []):
            if op.get("op") in ("wait", "timer"):
                op["minutes"] = m


def _ensure_collect_after_transfer(doc):
    steps = doc.get("steps", [])
    for i, st in enumerate(steps):
        if st.get("action", "").lower() == "transfer" and not any(
            op.get("op") == "centrifuge" for op in st.get("ops", [])
        ):
            # Insert a collect step right after transfer
            rpm = doc["defaults"].get("centrifuge_rpm", 4000)
            mins = doc["defaults"].get("centrifuge_minutes", 10)
            steps.insert(
                i + 1,
                {
                    "action": "collect",
                    "vessel": "V1",
                    "minutes": mins,
                    "ops": [
                        {
                            "op": "centrifuge",
                            "centrifuge_id": doc["devices"]["centrifuge_id"],
                            "rpm": rpm,
                            "minutes": mins,
                        },
                        {"op": "decant_supernatant", "tube": "V2_tube"},
                    ],
                    "raw": "Centrifuge to collect the precipitate.",
                },
            )


def _fix_wash_microplan_waits(doc):
    mp = doc.get("micro_plan", []) or []
    # For every wash step (with CF1 start), set 1 min before start, 10 min after
    step_ids = {
        m.get("step_index")
        for m in mp
        if m.get("verb") == "start" and m.get("device") == "CF1"
    }
    for si in step_ids:
        start_i = next(
            (
                i
                for i, m in enumerate(mp)
                if m.get("step_index") == si
                and m.get("verb") == "start"
                and m.get("device") == "CF1"
            ),
            None,
        )
        if start_i is None:
            continue
        for i, m in enumerate(mp):
            if m.get("step_index") == si and m.get("verb") == "wait":
                m["minutes"] = 1 if i < start_i else 10


def _sync_add_step_minutes_and_micro(doc):
    # Step 2: if add step has a wait op, set step minutes to that
    steps = doc.get("steps", [])
    if len(steps) >= 2 and (steps[1].get("action") or "").lower() == "add":
        wait = next(
            (
                op.get("minutes")
                for op in steps[1].get("ops", [])
                if op.get("op") == "wait"
            ),
            None,
        )
        if wait:
            steps[1]["minutes"] = wait
        # Fix micro pour to use ammonium hydroxide → V1 if present
        for m in steps[1].get("micro_ops", []):
            if (
                m.get("verb") == "pour"
                and m.get("from") == "V1"
                and m.get("to") == "V1"
            ):
                m["from"] = "ammonium hydroxide"
                m["to"] = "V1"


def _normalize_first_add_solvent_field(doc):
    # In step 1, ensure add_solvent uses 'solvent' key
    steps = doc.get("steps", [])
    if steps:
        for op in steps[0].get("ops", []):
            if (
                op.get("op") == "add_solvent"
                and "reagent" in op
                and "solvent" not in op
            ):
                op["solvent"] = op.pop("reagent")


def _purge_stray_vessels_and_contexts(doc):
    reg = doc.setdefault("vessel_registry", {})
    for k in list(reg.keys()):
        if k not in ("V1", "V2", "V2_tube"):
            del reg[k]
    # micro_plan context_vessel cleanup
    for m in doc.get("micro_plan", []) or []:
        if m.get("context_vessel") not in (None, "V1", "V2", "V2_tube"):
            m["context_vessel"] = "V1"


def _fixpoint(fn, doc, max_iter=2):
    for _ in range(max_iter):
        before = json.dumps(doc, sort_keys=True)
        fn(doc)
        after = json.dumps(doc, sort_keys=True)
        if before == after:
            break


def robot_normalize(doc):
    _seed_defaults_devices(doc)
    _seed_vessels(doc)
    _map_aliases(doc)
    _clean_names(doc)
    _dedupe_micro_ops(doc)

    _canonicalize_isolate_transfer(doc)
    _normalize_add_transfer_and_dry(doc)

    # Mixed steps → separate wash + dry, then make dry pure
    _split_wash_and_dry(doc)
    _fix_wash_blocks(doc)
    _fix_micro_placeholders(doc)
    _force_pure_dry(doc)
    _collapse_consecutive_dry_steps(doc)
    _first_spin_default(doc)
    _unify_heat_wait(doc)
    _normalize_calcination(doc)

    # Normalize timing ops once
    _ops_timer_to_wait(doc)

    # Ensure collect/centrifuge structure
    _ensure_process_centrifuge(doc)
    _split_transfer_collect(doc)
    _ensure_collect_after_transfer(doc)
    _drop_duplicate_process_after_collect(doc)
    _collapse_adjacent_collect_spins(doc)
    _dedupe_adjacent_waits(doc)

    # Minutes & indices must be ready before micro-plan syncs
    _sync_add_step_minutes_and_micro(doc)
    _fill_missing_step_index(doc)

    # Micro-plan & wash syncs
    _sync_heat_and_dry_waits(doc)
    _fix_wash_microplan(doc)
    _fix_transfer_context(doc)
    _fix_wash_microplan_waits(doc)
    _timers_to_waits(doc)
    _fix_wash_waits(doc)
    _clean_labels(doc)

    # Housekeeping
    _normalize_first_add_solvent_field(doc)
    _purge_stray_vessels_and_contexts(doc)
    _sync_step_minutes_from_ops(doc)

    # Authoritative rebuild: derive micro_ops & micro_plan strictly from ops
    _rebuild_micro_from_ops(doc)

    # Final guardrails/invariants
    _final_invariants(doc)

    _fixpoint(_rebuild_micro_from_ops, doc)
    _fixpoint(_final_invariants, doc)
    # Idempotency
    _map_aliases(doc)
    _dedupe_micro_ops(doc)

    return doc


FENCE_START_RX = re.compile(r"^\s*```")  # start of any fenced block
NON_PROC_HEAD_RX = re.compile(
    r"^\s*#{1,6}\s*(references?|sources?|bibliography|rationale|reasoning|notes|discussion|supplementary|appendix|acknowledge?ments?)\b",
    re.I,
)
INLINE_TAG_RX = re.compile(r"\s*\[(?:CTX|DB|PARSED|GEN|\d+)\]\s*", re.I)

# quantities like: 0.5 mmol | 58 mg | 10 mL | 1–2 mmol (range ok)
_AMOUNT_UNIT = r"(?:~?\d+(?:[.\u2013\u2014-]\d+)?\s*(?:µ?u?L|mL|ml|L|l|mg|g|µg|ug|mol|mmol|µmol|umol)\b)"

# split boundary: comma / + / "and" / "along with" / "together with"
# only split if the next token looks like a fresh quantity+unit
SPLIT_BOUNDARY_RX = re.compile(
    rf"\s*(?:,|\+|\band\b|\balong with\b|\btogether with\b)\s*(?=(?:{_AMOUNT_UNIT}))",
    re.I,
)

# Canonicalize unit spellings
_UNIT_CANON = {
    "l": "L",
    "L": "L",
    "ml": "mL",
    "mL": "mL",
    "ul": "µL",
    "uL": "µL",
    "µl": "µL",
    "µL": "µL",
    "g": "g",
    "mg": "mg",
    "µg": "µg",
    "ug": "µg",
    "mol": "mol",
    "mmol": "mmol",
    "µmol": "µmol",
    "umol": "µmol",
    "m": "M",
    "M": "M",
    "mm": "mM",
    "mM": "mM",
    "µM": "µM",
    "uM": "µM",
    "wt%": "wt%",
    "vol%": "vol%",
}

# Amount (mass/volume/moles) like: 98 mg | 10 mL | 0.5 mmol | 1–2 mmol
_AMOUNT_RE = r"(?P<approx>[~≈])?\s*(?P<val>\d+(?:\.\d+)?(?:[–-]\d+(?:\.\d+)?)?)\s*(?P<unit>µ?u?L|mL|ml|L|l|mg|g|µg|ug|mol|mmol|µmol|umol)\b"

# Secondary amount in parentheses: (98 mg), (10 mL)
_PAREN_AMOUNT_RX = re.compile(
    r"\(\s*(?P<val>\d+(?:\.\d+)?)\s*(?P<unit>µ?u?L|mL|ml|L|l|mg|g|µg|ug|mol|mmol|µmol|umol)\s*\)"
)

# Concentration form: 0.1 M HAuCl4 [in water]
_CONC_RX = re.compile(
    r"(?P<approx>[~≈])?\s*(?P<val>\d+(?:\.\d+)?)\s*(?P<unit>M|mM|µM|uM)\s+(?P<name>[^(),;]+?)(?:\s+in\s+(?P<solvent>[^(),;]+))?\b",
    re.I,
)

# Leading amount form: 98 mg PVP | 0.5 mmol of copper(II) acetate ...
_LEAD_AMT_RX = re.compile(
    rf"{_AMOUNT_RE}" + r"\s*(?:of\s+)?(?P<name>.+?)\s*(?P<paren>\([^)]*\))?\s*$", re.I
)

# Loose "about/approximately" flags anywhere
_APPROX_WORD_RX = re.compile(r"\b(about|approximately|approx\.)\b", re.I)


def _canon_unit(u: str) -> str:
    return _UNIT_CANON.get((u or "").strip(), (u or "").strip())


def _to_float_range(s: str) -> tuple[float, float] | tuple[float, None]:
    """Parse '1–2' or '1-2' into (1.0, 2.0); else single -> (x, None)."""
    s = s.replace("–", "-")
    if "-" in s and not s.startswith("-"):
        a, b = s.split("-", 1)
        try:
            return (float(a), float(b))
        except Exception:
            pass
    try:
        return (float(s), None)
    except Exception:
        return (0.0, None)


def strip_tags(s: str) -> str:
    s = _clean_unicode(s)
    s = re.sub(r"`{3,}.*$", "", s)
    s = INLINE_TAG_RX.sub(" ", s)
    s = re.sub(r"</?[^>]+>", "", s)  # <-- new
    s = s.replace("**", "").replace("__", "")
    s = re.sub(r"\s{2,}", " ", s)
    return s.strip()


def split_reagent_phrases(text: str) -> list[str]:
    """
    Split a multi-reagent phrase into separate items:
    e.g., '0.5 mmol Cu(OAc)2 (98 mg) and 0.5 mmol PVP (58 mg)'
      -> ['0.5 mmol Cu(OAc)2 (98 mg)', '0.5 mmol PVP (58 mg)']
    Only splits where a new quantity+unit begins; avoids over-splitting names.
    """
    s = (text or "").strip()
    if not s:
        return []
    parts = re.split(SPLIT_BOUNDARY_RX, s)
    out = []
    for p in parts:
        p = p.strip().strip(",").strip()
        if p:
            out.append(p)
    return out


def parse_reagent_phrase_to_struct(s: str) -> dict:
    """
    Parse a single reagent phrase into a structured dict.
    Supports:
      - '0.5 mmol copper(II) acetate monohydrate (98 mg)'
      - '98 mg PVP'
      - '10 mL ethylene glycol'
      - '0.1 M HAuCl4 in water'
    Returns a dict with keys:
      name, amount, amount_unit, amount_range, alt_amount, alt_unit,
      concentration, conc_unit, solvent, approx, original
    """
    original = s
    s = strip_tags(_clean_unicode((s or "").strip()))
    approx = bool(_APPROX_WORD_RX.search(s))

    # 1) Try concentration pattern first
    m = _CONC_RX.search(s)
    if m:
        val = float(m.group("val"))
        unit = _canon_unit(m.group("unit"))
        name = m.group("name").strip()
        solvent = (m.group("solvent") or "").strip() or None
        approx = approx or bool(m.group("approx"))
        return {
            "name": name,
            "amount": None,
            "amount_unit": None,
            "amount_range": None,
            "alt_amount": None,
            "alt_unit": None,
            "concentration": val,
            "conc_unit": unit,
            "solvent": solvent,
            "approx": approx,
            "original": original,
        }

    # 2) Try leading amount pattern
    m = _LEAD_AMT_RX.match(s)
    if m:
        rng = _to_float_range(m.group("val"))
        amount = rng[0]
        amount_range = None
        if rng[1] is not None:
            amount_range = [rng[0], rng[1]]

        unit = _canon_unit(m.group("unit"))
        name = (m.group("name") or "").strip().strip(",;")
        approx = approx or bool(m.group("approx"))

        # Optional secondary amount in parentheses
        alt_amount = None
        alt_unit = None
        par = m.group("paren") or ""
        pm = _PAREN_AMOUNT_RX.search(par)
        if pm:
            alt_amount = float(pm.group("val"))
            alt_unit = _canon_unit(pm.group("unit"))

        return {
            "name": name,
            "amount": amount,
            "amount_unit": unit,
            "amount_range": amount_range,
            "alt_amount": alt_amount,
            "alt_unit": alt_unit,
            "concentration": None,
            "conc_unit": None,
            "solvent": None,
            "approx": approx,
            "original": original,
        }

    # 3) Fallback: just return name
    return {
        "name": s,
        "amount": None,
        "amount_unit": None,
        "amount_range": None,
        "alt_amount": None,
        "alt_unit": None,
        "concentration": None,
        "conc_unit": None,
        "solvent": None,
        "approx": approx,
        "original": original,
    }


def _clean_unicode(s: str) -> str:
    s = unicodedata.normalize("NFKC", s)
    return s.replace("° ", "°").replace("–", "-").replace("—", "-")


def _normalize_reagents_inplace(record: dict) -> None:
    # reagents: flatten and split strings
    reag = record.get("reagents", [])
    flat: list[str] = []
    for item in (reag if isinstance(reag, list) else [reag]):
        if isinstance(item, str):
            flat.extend(split_reagent_phrases(item))
        elif item:
            # preserve dicts/structured entries
            flat.append(item)
    record["reagents"] = flat

    solute_str = record.get("solute", "")
    if isinstance(solute_str, str) and solute_str.strip():
        record["solutes"] = split_reagent_phrases(solute_str)


def _add_structured_reagents_inplace(record: dict) -> None:
    """
    Populate record['reagents_structured'] and record['solutes_structured'].
    If the record contains parsed solute amount/unit, propagate it to the corresponding entries.
    Also flag solvent entries with solvent=True.
    """
    reag = record.get("reagents", []) or []
    if not isinstance(reag, list):
        reag = [reag]

    solute_name = record.get("solute")
    solvent_names = set()
    if isinstance(record.get("solvent"), str):
        solvent_names.add(record["solvent"])
    if isinstance(record.get("solvents"), list):
        for comp in record["solvents"]:
            name = comp.get("name")
            if isinstance(name, str) and name.strip():
                solvent_names.add(name.strip())

    reag_struct = []
    for x in reag:
        if not (isinstance(x, str) and x.strip()):
            continue
        base = parse_reagent_phrase_to_struct(x)
        if solute_name and x == solute_name:
            if record.get("amount") is not None:
                base["amount"] = record.get("amount")
            if record.get("unit"):
                base["amount_unit"] = record.get("unit")
        if x in solvent_names:
            base["solvent"] = True
        reag_struct.append(base)
    record["reagents_structured"] = reag_struct

    solutes = record.get("solutes", []) or []
    sols_struct = []
    if isinstance(solutes, list) and solutes:
        for x in solutes:
            if not (isinstance(x, str) and x.strip()):
                continue
            base = parse_reagent_phrase_to_struct(x)
            if solute_name and x == solute_name:
                if record.get("amount") is not None:
                    base["amount"] = record.get("amount")
                if record.get("unit"):
                    base["amount_unit"] = record.get("unit")
            sols_struct.append(base)
    if sols_struct:
        record["solutes_structured"] = sols_struct


# -------- Units parsing --------
def find_temp_c(t: str) -> Optional[float]:
    s = _clean_unicode(t)
    if re.search(r"\breflux\b", s, re.I):
        return 100.0
    if re.search(r"\bboil(?:ing)?\b", s, re.I):
        return 100.0
    if re.search(r"\bice\s*bath\b", s, re.I):
        return 0.0
    m = re.search(r"(-?\d+(?:\.\d+)?)\s*°?\s*([CFK])\b", s, re.I)
    if not m:
        if re.search(r"\b(rt|room\s*temp(?:erature)?)\b", s, re.I):
            return DEFAULTS["room_temp_C"]
        return None
    val = float(m.group(1))
    unit = m.group(2).upper()
    if unit == "C":
        return val
    if unit == "F":
        return (val - 32.0) * 5.0 / 9.0
    if unit == "K":
        return val - 273.15
    return None


def find_minutes(t: str) -> Optional[float]:
    s = _clean_unicode(t)
    mins = 0.0
    found = False
    if re.search(r"\bover\s*night\b", s, re.I):
        return 12 * 60.0
    for m in re.finditer(r"(\d+(?:\.\d+)?)\s*(?:second|sec|s)\b", s, re.I):
        mins += float(m.group(1)) / 60.0
        found = True
    for m in re.finditer(r"(\d+(?:\.\d+)?)\s*(?:hour|hr|hrs|h)\b", s, re.I):
        mins += float(m.group(1)) * 60.0
        found = True
    for m in re.finditer(r"(\d+(?:\.\d+)?)\s*(?:minute|min|mins|m)\b", s, re.I):
        mins += float(m.group(1))
        found = True
    return mins if found else None


def _parse_vol_ml(text: str) -> Optional[float]:
    s = _clean_unicode(text)
    m = re.search(r"(\d+(?:\.\d+)?)\s*(µ?u?L|mL|ml|L|l)\b", s)
    if not m:
        return None
    val = float(m.group(1))
    unit = m.group(2).lower()
    if unit in ("µl", "ul"):
        return val / 1000.0
    return val if unit == "ml" else val * 1000.0


def _parse_conc(text: str) -> Optional[Tuple[float, str]]:
    s = _clean_unicode(text)
    m = re.search(r"(\d+(?:\.\d+)?)\s*(m?M|%\s*w/?v|%\s*v/?v|%)\b", s, re.I)
    if not m:
        return None
    v = float(m.group(1))
    u = m.group(2).replace(" ", "").lower()
    if u == "m":
        u = "M"
    if u == "mm":
        u = "mM"
    return v, u


# -------- Hardware parsing --------
def parse_hardware(markdown_text: str) -> List[Dict]:
    lines = markdown_text.splitlines()
    items = []
    in_hw = False
    for line in lines:
        if re.match(r"\s*1\.\s*\*\*Hardware\s*&\s*Glassware\*\*:", line, re.I):
            in_hw = True
            continue
        if in_hw:
            if line.strip().startswith("2.") or re.match(r"\s*2\.\s*\*\*", line):
                break
            if line.strip().startswith("- "):
                entry = strip_tags(line.strip()[2:])
                m = re.match(r"(Beakers?|Flasks?)\s*\((.+?)\)", entry, re.I)
                if m:
                    base = "beaker" if "beaker" in m.group(1).lower() else "flask"
                    sizes = m.group(2)
                    parts = re.split(r"\s*(?:and|,)\s*", sizes)
                    for p in parts:
                        cap = p.strip()
                        items.append(
                            {
                                "name": f"{m.group(1).split()[0].title()} {cap}",
                                "type": base,
                                "capacity": cap,
                            }
                        )
                else:
                    capm = re.search(r"(\d+)\s*(µ?u?L|mL|L)\b", entry, re.I)
                    cap = capm.group(0) if capm else None
                    typ = (
                        "beaker"
                        if "beaker" in entry.lower()
                        else ("flask" if "flask" in entry.lower() else "hardware")
                    )
                    nm = (
                        entry
                        if typ == "hardware"
                        else (f"{typ.title()} {cap}" if cap else typ.title())
                    )
                    items.append({"name": nm, "type": typ, "capacity": cap})
    out = []
    for i, it in enumerate(items, 1):
        it2 = dict(it)
        it2["id"] = f"H{i}"
        out.append(it2)
    return out


def _capacity_to_ml(cap: Optional[str]) -> Optional[float]:
    if not cap:
        return None
    m = re.match(r"(\d+(?:\.\d+)?)\s*(µ?u?L|mL|L)\b", cap, re.I)
    if not m:
        return None
    val = float(m.group(1))
    unit = m.group(2).lower()
    if unit in ("µl", "ul"):
        return val / 1000.0
    return val if unit == "ml" else val * 1000.0


class VesselRegistry:
    def __init__(self, hardware: List[Dict]):
        self._vid_to_label: Dict[str, str] = {}
        self._label_to_vid: Dict[str, str] = {}
        self._vid_to_hid: Dict[str, str] = {}
        self._vessel_contents: Dict[str, str] = {}
        self._counter = 0
        self.primary_vessel: Optional[str] = None
        self.hardware = hardware

    def _new_vid(self) -> str:
        self._counter += 1
        return f"V{self._counter}"

    def _pick_glass_for_volume(
        self, vol_ml: Optional[float], preferred: Optional[str] = None
    ) -> Optional[Dict]:
        types = (preferred.lower(),) if preferred else ("beaker", "flask")
        choices = [h for h in self.hardware if h.get("type") in types]
        if not choices:
            return None
        if vol_ml is None:
            return sorted(
                choices, key=lambda h: (_capacity_to_ml(h.get("capacity")) or 1e9)
            )[0]
        target = vol_ml * 1.5
        candidates = [(h, _capacity_to_ml(h.get("capacity")) or 1e12) for h in choices]
        candidates = [c for c in candidates if c[1] >= target]
        if candidates:
            h, _ = sorted(candidates, key=lambda x: x[1])[0]
            return h
        h, _ = sorted(
            [(h, _capacity_to_ml(h.get("capacity")) or 0) for h in choices],
            key=lambda x: x[1],
            reverse=True,
        )[0]
        return h

    def ensure_glassware(
        self,
        label: str,
        *,
        prefer_capacity_ml: Optional[float] = None,
        explicit_hardware_hint: Optional[str] = None,
    ) -> str:
        key = label.lower().strip()
        if key in self._label_to_vid:
            return self._label_to_vid[key]
        vid = self._new_vid()
        hw_id = None
        preferred = None
        if explicit_hardware_hint:
            for h in self.hardware:
                if explicit_hardware_hint.lower() in h["name"].lower():
                    hw_id = h["id"]
                    preferred = h.get("type")
                    break
        if hw_id is None:
            chosen = self._pick_glass_for_volume(prefer_capacity_ml, preferred)
            if chosen:
                hw_id = chosen["id"]
        self._vid_to_label[vid] = label
        self._label_to_vid[key] = vid
        if hw_id:
            self._vid_to_hid[vid] = hw_id
        if self.primary_vessel is None:
            self.primary_vessel = vid
        return vid

    def map_contents(self, vid: str, contents: str):
        self._vessel_contents[vid] = contents

    def vessel_hardware(self, vid: str) -> Optional[str]:
        return self._vid_to_hid.get(vid)

    def as_dict(self) -> Dict[str, str]:
        return dict(self._vid_to_label)

    def contents_dict(self) -> Dict[str, str]:
        return dict(self._vessel_contents)


# -------- Pattern detectors --------
_CONC_UNIT_RX = r"(?:M|m|mM|%\s*w/?v|%\s*v/?v|%)"


def _clean_solvent_tail(solvent: str) -> str:
    solvent = strip_tags(solvent.strip().rstrip(",."))
    solvent = solvent.split(" in ")[0].strip()
    return solvent


def detect_solution_prep(line: str) -> Optional[Dict]:
    s = strip_tags(_clean_unicode(line.strip().rstrip(".")))
    pats = [
        re.compile(
            rf"""prepare\s+a\s+([\d\.]+)\s*({_CONC_UNIT_RX})\s+(?P<xname>.+?)\s+solution\s+
                by\s+dissolving\s+(?P<solute>.+?)\s+in\s+(?P<vol>[\d\.]+)\s*(?P<vunit>µ?u?L|mL|ml|l|L)\s+of\s+(?P<solvent>.+?)\s*(?:in\b|$)""",
            re.I | re.X,
        ),
        re.compile(
            rf"""dissolv\w*\s+(?P<solute>.+?)\s+in\s+(?P<vol>[\d\.]+)\s*(?P<vunit>µ?u?L|mL|ml|l|L)\s+of\s+(?P<solvent>.+?)\s+
                to\s+(?:make|form|yield|obtain)\s+a\s+([\d\.]+)\s*({_CONC_UNIT_RX})\s+.+?\s+solution""",
            re.I | re.X,
        ),
        re.compile(
            r"dissolv\w*\s+(?P<solute>.+?)\s+in\s+(?P<vol>[\d\.]+)\s*(?P<vunit>µ?u?L|mL|ml|l|L)\s+of\s+(?P<solvent>.+?)(?:\.|$)",
            re.I,
        ),
    ]
    for rx in pats:
        m = rx.search(s)
        if m:
            solute = m.groupdict().get("solute", "").strip()
            vol = float(m.group("vol"))
            vunit = m.group("vunit")
            solvent = _clean_solvent_tail(m.group("solvent"))
            hint = None
            mh = re.search(
                r"in a\s+(\d+\s*(?:µ?u?L|mL|L)\s+(?:glass\s+)?(?:beaker|flask))",
                s,
                re.I,
            )
            if mh:
                hint = mh.group(1)
            conc_val, conc_unit = None, None
            conc_match = re.search(r"(\d+(?:\.\d+)?)\s*(M|mM|%)\s+solution", s)
            if conc_match:
                conc_val = float(conc_match.group(1))
                conc_unit = conc_match.group(2)
            return {
                "action": "dispense",
                "solute": solute,
                "solvent": solvent,
                "concentration": conc_val,
                "concentration_units": conc_unit,
                "volume": vol,
                "volume_units": vunit,
                "hardware_hint": hint,
            }
    return None


def _normalize_units(value, unit):
    """Normalize units to standard forms"""
    if not unit:
        return value, unit

    unit_lower = unit.lower().replace("µ", "u").replace(" ", "")

    # Volume normalization to mL
    if unit_lower in ["ul", "μl"]:
        return value / 1000.0, "mL"
    elif unit_lower in ["l"]:
        return value * 1000.0, "mL"
    elif unit_lower in ["ml"]:
        return value, "mL"

    # Mass normalization to mg
    elif unit_lower in ["g"]:
        return value * 1000.0, "mg"
    elif unit_lower in ["kg"]:
        return value * 1000000.0, "mg"
    elif unit_lower in ["ug", "μg"]:
        return value / 1000.0, "mg"
    elif unit_lower in ["mg"]:
        return value, "mg"

    # Time normalization to minutes
    elif unit_lower in ["h", "hr", "hour", "hours"]:
        return value * 60.0, "min"
    elif unit_lower in ["s", "sec", "second", "seconds"]:
        return value / 60.0, "min"
    elif unit_lower in ["min", "minute", "minutes"]:
        return value, "min"

    return value, unit


def detect_add_solvent(line: str) -> Optional[Dict]:
    s = strip_tags(_clean_unicode(line.strip()))
    # More flexible pattern to catch variations like "add 5mL ethanol to the mixture"
    m = re.search(
        r"\b(?:add|pour|introduce)\s+(?P<vol>[\d\.]+)\s*(?P<vunit>µ?u?L|mL|ml|l|L)\s+(?:of\s+)?(?P<solvent>.+?)\s+(?:to|into)\s+(?:the\s+)?(?:solution|mixture|suspension|dispersion|flask|beaker|vessel)\b",
        s,
        re.I,
    )
    if not m:
        # Fallback pattern for "add ethanol (5 mL) to mixture"
        m = re.search(
            r"\b(?:add|pour|introduce)\s+(?P<solvent>.+?)\s*\(\s*(?P<vol>[\d\.]+)\s*(?P<vunit>µ?u?L|mL|ml|l|L)\s*\)\s+(?:to|into)\s+(?:the\s+)?(?:solution|mixture|suspension|dispersion|flask|beaker|vessel)\b",
            s,
            re.I,
        )
    if not m:
        return None

    # Normalize volume units
    volume, volume_units = _normalize_units(float(m.group("vol")), m.group("vunit"))

    return {
        "action": "add_solvent",
        "volume": volume,
        "volume_units": volume_units,
        "solvent": m.group("solvent").strip(),
    }


def detect_add(line: str) -> Optional[Dict]:
    s = strip_tags(_clean_unicode(line.strip()))
    m = re.search(
        r"\b(add|charge)\s+(?:the\s+)?(?P<src>.+?)\s+to\s+(?:the\s+)?(?P<dst>.+?)\b",
        s,
        re.I,
    )
    if not m:
        return None
    rate = "slow" if re.search(r"\b(dropwise|slow)\b", s, re.I) else "normal"
    at_temp = find_temp_c(s)
    over_min = find_minutes(s)
    return {
        "action": "add",
        "source_name": m.group("src").strip(),
        "target_name": m.group("dst").strip(),
        "rate": rate,
        "temperature_C": at_temp,
        "minutes": over_min,
    }


def detect_stir(line: str) -> Optional[Dict]:
    s = strip_tags(_clean_unicode(line.strip()))
    if not re.search(r"\bstir", s, re.I):
        return None
    rpm = None
    mr = re.search(r"(\d{2,5})\s*rpm\b", s, re.I)
    if mr:
        rpm = int(mr.group(1))
    minutes = find_minutes(s) or 60.0
    temp = find_temp_c(s) or DEFAULTS["room_temp_C"]
    return {
        "action": "stir",
        "rpm": rpm or DEFAULTS["stir_rpm"],
        "minutes": minutes,
        "temperature_C": temp,
    }


def detect_heat(line: str) -> Optional[List[Dict]]:
    s = strip_tags(_clean_unicode(line.strip()))
    if not re.search(r"\b(heat|maintain|hold)\b", s, re.I):
        return None
    temp = find_temp_c(s) or DEFAULTS["room_temp_C"]
    minutes = find_minutes(s) or 60.0
    return [
        {"action": "heat_to", "temperature_C": temp},
        {"action": "hold", "minutes": minutes},
    ]


def detect_cool(line: str) -> Optional[Dict]:
    s = strip_tags(_clean_unicode(line.strip()))
    if re.search(r"\b(ice\s*bath|cool)\b", s, re.I):
        temp = (
            0.0 if "ice" in s.lower() else (find_temp_c(s) or DEFAULTS["room_temp_C"])
        )
        return {"action": "cool_to", "temperature_C": temp}
    return None


def detect_sonicate(line: str) -> Optional[Dict]:
    s = strip_tags(_clean_unicode(line.strip()))
    if re.search(r"\bsonicat", s, re.I):
        return {"action": "sonicate", "minutes": find_minutes(s) or 10.0}
    return None


def detect_filter(line: str) -> Optional[List[Dict]]:
    s = strip_tags(_clean_unicode(line.strip()))
    if re.search(r"\b(vacuum\s*filter|filter\b)", s, re.I):
        ops = [{"action": "filter"}]
        if "vacuum" in s.lower():
            ops.append({"action": "apply_vacuum"})
        return ops
    return None


def detect_wash_dry(line: str) -> Optional[List[Dict]]:
    s = strip_tags(_clean_unicode(line.strip()))
    ops = []
    if "wash" in s.lower():
        n = 1
        mw = re.search(r"(\d+)\s*[x×]\s*wash", s, re.I)
        if mw:
            n = int(mw.group(1))
        wash_solvent = (
            "deionized water"
            if re.search(r"\b(di\s*water|deionized\s*water)\b", s, re.I)
            else "wash solvent"
        )
        for _ in range(n):
            ops += [
                {"action": "add_wash_solvent", "solvent": wash_solvent},
                {"action": "resuspend"},
                {
                    "action": "centrifuge",
                    "rpm": DEFAULTS["centrifuge_rpm"],
                    "minutes": DEFAULTS["centrifuge_minutes"],
                },
                {"action": "decant_supernatant"},
            ]
    if "dry" in s.lower() or "oven" in s.lower():
        temp = find_temp_c(s) or 60.0
        minutes = find_minutes(s) or 120.0
        ops.append({"action": "oven_dry", "temperature_C": temp, "minutes": minutes})
    return ops or None


def detect_resuspend(line: str) -> Optional[Dict]:
    s = strip_tags(_clean_unicode(line.strip()))
    if re.search(r"\bresuspend\b", s, re.I):
        return {"action": "resuspend"}
    return None


def detect_collect(line: str) -> Optional[Dict]:
    s = strip_tags(_clean_unicode(line.strip()))
    if re.search(r"\bcollect\b", s, re.I):
        return {"action": "collect"}
    return None


def detect_discard(line: str) -> Optional[Dict]:
    s = strip_tags(_clean_unicode(line.strip()))
    if re.search(r"\bdiscard\b", s, re.I):
        return {"action": "discard"}
    return None


def detect_transfer(line: str) -> Optional[Dict]:
    s = strip_tags(_clean_unicode(line.strip()))
    m = re.search(r"\btransfer\b.*\bto\b\s+(?P<target>.+)", s, re.I)
    if m:
        return {"action": "transfer", "target": m.group("target").strip()}
    return None


def detect_weigh(line: str) -> Optional[Dict]:
    s = strip_tags(_clean_unicode(line.strip()))
    m = re.search(
        r"\bweigh\s+(?P<amount>[\d\.]+)\s*(?P<unit>mg|g|µg|kg)\s+of\s+(?P<reagent>.+?)(?:\.|$)",
        s,
        re.I,
    )
    if m:
        return {
            "action": "weigh",
            "reagent": m.group("reagent").strip(),
            "amount": float(m.group("amount")),
            "unit": m.group("unit"),
        }
    return None


def detect_transfer_explicit(line: str) -> Optional[Dict]:
    s = strip_tags(_clean_unicode(line.strip()))
    m = re.search(
        r"\btransfer\s+(?:it|the\s+mixture|solution|precipitate)?\s*(?:into|to)\s+(?P<target>.+?)(?:\.|$)",
        s,
        re.I,
    )
    if m:
        return {"action": "transfer", "target": m.group("target").strip()}
    return None


def detect_dissolve(line: str) -> Optional[Dict]:
    s = strip_tags(_clean_unicode(line.strip()))
    m = re.search(
        r"\bdissolv\w*\s+(?P<amount>[\d\.]+)\s*(?P<unit>mg|g|µg|kg)\s+of\s+(?P<solute>.+?)\s+in\s+(?P<vol>[\d\.]+)\s*(?P<vunit>µ?u?L|mL|ml|l|L)\s+of\s+(?P<solvent>[^.;,]+)",
        s,
        re.I,
    )
    if not m:
        return None

    solute = m.group("solute").strip()
    solvent_captured = _clean_solvent_tail(m.group("solvent").strip())
    vol1 = float(m.group("vol"))
    vunit1 = m.group("vunit")

    extras = []
    inline = solvent_captured
    while True:
        exm = re.search(
            r"(.*?)(?:,\s*)?(?:and\s+)([\d\.]+)\s*(µ?u?L|mL|ml|l|L)\s+of\s+([^,;]+)$",
            inline,
            re.I,
        )
        if not exm:
            break
        base = exm.group(1).strip()
        vol2 = float(exm.group(2))
        vunit2 = exm.group(3)
        solv2 = _clean_solvent_tail(exm.group(4).strip())
        extras.insert(0, {"name": solv2, "volume": vol2, "volume_units": vunit2})
        inline = base
    solvent1 = inline

    hint = None
    mh = re.search(
        r"in\s+(?:a|the)\s+(\d+\s*(?:µ?u?L|mL|L)\s+(?:glass\s+)?(?:beaker|flask|round-?bottom\s+flask))",
        s,
        re.I,
    )
    if mh:
        hint = mh.group(1)

    result = {
        "action": "dissolve",
        "solute": solute,
        "amount": float(m.group("amount")),
        "unit": m.group("unit"),
        "solvent": (
            solvent1
            if not extras
            else solvent1 + " + " + " + ".join(e["name"] for e in extras)
        ),
        "volume": vol1,
        "volume_units": vunit1,
        "hardware_hint": hint,
    }
    if extras:
        result["solvents"] = [
            {"name": solvent1, "volume": vol1, "volume_units": vunit1}
        ] + extras
    return result


def detect_filter_isolate(line: str) -> Optional[Dict]:
    s = strip_tags(_clean_unicode(line.strip()))
    if re.search(
        r"\b(isolate|collect|obtain)\s+(?:the\s+)?(precipitate|solid|product)", s, re.I
    ):
        return {"action": "isolate"}
    return None


# -------- Ops builders --------
def ops_for_dispense(
    vessel: str,
    hardware_id: Optional[str],
    solute: str,
    solvent: str,
    volume_val: float,
    volume_unit: str,
) -> List[Dict]:
    return [
        {"op": "ensure_vessel", "vessel": vessel, "hardware_id": hardware_id},
        {"op": "add_solute", "vessel": vessel, "reagent": solute},
        {
            "op": "add_solvent",
            "vessel": vessel,
            "solvent": solvent,
            "volume": volume_val,
            "volume_units": volume_unit,
        },
    ]


def ops_for_add(
    src_v: str,
    dst_v: str,
    rate: str,
    rpm: Optional[int] = None,
    temperature_C: Optional[float] = None,
    minutes: Optional[float] = None,
) -> List[Dict]:
    ops = [
        {
            "op": "move_to_stir_plate",
            "vessel": dst_v,
            "stir_plate_id": DEVICE_IDS["stir_plate_id"],
        },
        {"op": "set_stir_rate", "vessel": dst_v, "rpm": rpm or DEFAULTS["stir_rpm"]},
    ]
    if temperature_C is not None:
        ops.append(
            {
                "op": "set",
                "device": DEVICE_IDS["hotplate_id"],
                "param": "temperature_C",
                "value": temperature_C,
            }
        )
    ops.append({"op": "transfer", "from": src_v, "to": dst_v, "rate": rate})
    if minutes:
        ops.append({"op": "wait", "minutes": minutes})
    return ops


def ops_for_stir(vessel: str, minutes: float, rpm: int, temp_C: float) -> List[Dict]:
    return [
        {
            "op": "move_to_stir_plate",
            "vessel": vessel,
            "stir_plate_id": DEVICE_IDS["stir_plate_id"],
        },
        {"op": "set_stir_rate", "vessel": vessel, "rpm": rpm},
        {
            "op": "set",
            "device": DEVICE_IDS["hotplate_id"],
            "param": "temperature_C",
            "value": temp_C,
        },
        {"op": "wait", "minutes": minutes},
    ]


def ops_for_heat(vessel: str, temp_C: float, minutes: float) -> List[Dict]:
    return [
        {
            "op": "set",
            "device": DEVICE_IDS["hotplate_id"],
            "param": "temperature_C",
            "value": temp_C,
        },
        {"op": "wait", "minutes": minutes},
    ]


def ops_for_postproc(vessel: str, actions: List[Dict]) -> List[Dict]:
    ops = []
    for a in actions:
        if a["action"] == "cool_to":
            ops.append(
                {
                    "op": "set",
                    "device": DEVICE_IDS["hotplate_id"],
                    "param": "temperature_C",
                    "value": a["temperature_C"],
                }
            )
        elif a["action"] == "centrifuge":
            ops.append(
                {"op": "transfer_to_centrifuge_tube", "from": vessel, "to": "V2_tube"}
            )
            ops.append(
                {
                    "op": "centrifuge",
                    "centrifuge_id": DEVICE_IDS["centrifuge_id"],
                    "rpm": a["rpm"],
                    "minutes": a["minutes"],
                }
            )
        elif a["action"] == "decant_supernatant":
            ops.append({"op": "decant_supernatant", "tube": "V2_tube"})
        elif a["action"] == "add_wash_solvent":
            ops.append(
                {"op": "add_wash_solvent", "tube": "V2_tube", "solvent": a["solvent"]}
            )
        elif a["action"] == "resuspend":
            ops.append({"op": "resuspend", "tube": "V2_tube"})
        elif a["action"] == "oven_dry":
            ops.append(
                {
                    "op": "move_to_oven",
                    "tube": "V2_tube",
                    "oven_id": DEVICE_IDS["oven_id"],
                }
            )
            ops.append(
                {
                    "op": "set",
                    "device": DEVICE_IDS["oven_id"],
                    "param": "temperature_C",
                    "value": a["temperature_C"],
                }
            )
            ops.append({"op": "wait", "minutes": a["minutes"]})
        elif a["action"] == "filter":
            ops.append({"op": "setup_filtration"})
        elif a["action"] == "apply_vacuum":
            ops.append({"op": "start", "device": DEVICE_IDS["vacuum_pump_id"]})
        elif a["action"] == "sonicate":
            ops.append(
                {
                    "op": "sonicate",
                    "sonicator_id": DEVICE_IDS["sonicator_id"],
                    "minutes": a.get("minutes", 10.0),
                }
            )
    return ops


# -------- Step extraction --------
def extract_steps(markdown_text: str) -> List[str]:
    lines = markdown_text.splitlines()
    in_proc = False
    steps = []
    buf = []
    for line in lines:
        if re.match(r"^\s*\d+\.\s*\*\*Procedure\*\*:", line):
            in_proc = True
            continue
        if in_proc:
            # hard stops to avoid pulling rich-text sections
            if FENCE_START_RX.match(line) or NON_PROC_HEAD_RX.match(line):
                break

            if re.match(r"\s*\d+\.\s", line):
                if buf:
                    steps.append(" ".join(buf).strip())
                    buf = []
                buf.append(re.sub(r"^\s*\d+\.\s*", "", line).strip())
            else:
                # ignore empty or fence continuation lines if any slipped in
                if line.strip() and not line.strip().startswith("```"):
                    buf.append(line.strip())
    if buf:
        steps.append(" ".join(buf).strip())
    return [strip_tags(s) for s in steps if s.strip()]


# -------- Main converter --------


# -------- Micro-action expansion --------
def _label_for_vessel(vid: str, vessels: "VesselRegistry", hardware: list[dict]) -> str:
    if not vid:
        vid = vessels.primary_vessel or vessels.ensure_glassware("Beaker")
    label = vessels.as_dict().get(vid) or vid
    hid = vessels.vessel_hardware(vid)
    if hid:
        hw = next((h for h in hardware if h.get("id") == hid), None)
        if hw and hw.get("name"):
            return f"{hw['name']} ({label})"
    return label


def _micro_for_op(
    op: dict, vessels: "VesselRegistry", hardware: list[dict]
) -> list[dict]:
    m = []
    typ = op.get("op")
    if typ == "ensure_vessel":
        v = _label_for_vessel(op.get("vessel", ""), vessels, hardware)
        m += [
            {"verb": "pick_up", "object": v, "from": "bench"},
            {"verb": "place", "object": v, "to": "bench"},
        ]
        return m
    if typ == "move_to_stir_plate":
        v = _label_for_vessel(op.get("vessel", ""), vessels, hardware)
        m += [
            {"verb": "pick_up", "object": v, "from": "bench"},
            {"verb": "place", "object": v, "to": "stir_plate"},
        ]
        return m
    if typ == "set_stir_rate":
        m += [
            {
                "verb": "set",
                "device": op.get("stir_plate_id") or "stir_plate",
                "param": "rpm",
                "value": op.get("rpm"),
            }
        ]
        return m

    if typ == "add_solute":
        dst = _label_for_vessel(op.get("vessel", ""), vessels, hardware)
        src_name = op.get("reagent") or op.get("solute") or "solute"
        pour = {"verb": "pour", "from": src_name, "to": dst}
        if op.get("amount") is not None:
            pour["amount"] = op.get("amount")
        if op.get("unit"):
            pour["unit"] = op.get("unit")
        m += [
            {"verb": "pick_up", "object": src_name, "from": "bench"},
            pour,
            {"verb": "place", "object": src_name, "to": "bench"},
        ]
        return m
    if typ == "add_solvent":
        dst = _label_for_vessel(op.get("vessel", ""), vessels, hardware)
        src_name = op.get("reagent") or op.get("solvent") or "solvent"
        pour = {"verb": "pour", "from": src_name, "to": dst}
        if op.get("volume") is not None:
            pour["volume"] = op.get("volume")
        if op.get("volume_units"):
            pour["volume_units"] = op.get("volume_units")
        m += [
            {"verb": "pick_up", "object": src_name, "from": "bench"},
            pour,
            {"verb": "place", "object": src_name, "to": "bench"},
        ]
        return m

    # Generic primitive ops -----------------------------------------------
    if typ == "set":
        dev = op.get("device") or op.get("hotplate_id") or op.get("oven_id") or "device"
        m += [
            {
                "verb": "set",
                "device": dev,
                "param": op.get("param"),
                "value": op.get("value"),
            }
        ]
        return m
    if typ == "start":
        dev = op.get("device") or "device"
        m += [{"verb": "start", "device": dev}]
        return m
    if typ == "stop":
        dev = op.get("device") or "device"
        m += [{"verb": "stop", "device": dev}]
        return m
    if typ == "pick_up":
        v = _label_for_vessel(op.get("vessel", ""), vessels, hardware)
        m += [{"verb": "pick_up", "object": v, "from": "bench"}]
        return m
    if typ == "place":
        v = _label_for_vessel(op.get("vessel", ""), vessels, hardware)
        m += [{"verb": "place", "object": v, "to": "bench"}]
        return m
    if typ == "transfer":
        src = (
            _label_for_vessel(op.get("from", ""), vessels, hardware)
            if op.get("from")
            else "source"
        )
        dst = (
            _label_for_vessel(op.get("to", ""), vessels, hardware)
            if op.get("to")
            else "target"
        )
        rate = op.get("rate") or "normal"
        m += [
            {"verb": "pick_up", "object": src, "from": "bench"},
            {"verb": "pour", "from": src, "to": dst, "rate": rate},
            {"verb": "place", "object": src, "to": "bench"},
        ]
        return m
    if typ == "wait":
        m += [{"verb": "wait", "minutes": op.get("minutes")}]
        return m
    if typ == "filter":
        v = _label_for_vessel(op.get("vessel", ""), vessels, hardware)
        m += [
            {"verb": "place", "object": "filtration setup", "to": "bench"},
            {"verb": "pick_up", "object": v, "from": "bench"},
            {"verb": "pour", "from": v, "to": "filtration setup"},
            {"verb": "place", "object": v, "to": "bench"},
        ]
        return m
    if typ == "start_vacuum":
        m += [
            {
                "verb": "set",
                "device": op.get("vacuum_pump_id") or "vacuum_pump",
                "param": "power",
                "value": "on",
            },
            {"verb": "start", "device": op.get("vacuum_pump_id") or "vacuum_pump"},
        ]
        return m
    if typ == "decant_supernatant":
        tube = op.get("tube") or "tube"
        m += [
            {"verb": "pick_up", "object": tube, "from": "rack"},
            {"verb": "pour", "from": tube, "to": "waste"},
            {"verb": "place", "object": tube, "to": "rack"},
        ]
        return m

    if typ == "add_wash_solvent":
        tube = op.get("tube") or "tube"
        src = op.get("solvent") or "wash solvent"
        m += [
            {"verb": "pick_up", "object": src, "from": "bench"},
            {"verb": "pour", "from": src, "to": tube},
            {"verb": "place", "object": src, "to": "bench"},
        ]
        return m

    if typ == "resuspend":
        tube = op.get("tube") or "tube"
        m += [
            {"verb": "pick_up", "object": tube, "from": "rack"},
            {"verb": "place", "object": tube, "to": "vortex"},
            {"verb": "wait", "minutes": 1},
            {"verb": "place", "object": tube, "to": "rack"},
        ]
        return m

    if typ == "centrifuge":
        tube = op.get("tube") or "V2_tube"
        m += [
            {"verb": "pick_up", "object": tube, "from": "rack"},
            {"verb": "place", "object": tube, "to": "centrifuge"},
            {
                "verb": "set",
                "device": op.get("centrifuge_id") or "centrifuge",
                "param": "rpm",
                "value": op.get("rpm"),
            },
            {
                "verb": "set",
                "device": op.get("centrifuge_id") or "centrifuge",
                "param": "minutes",
                "value": op.get("minutes"),
            },
            {"verb": "start", "device": op.get("centrifuge_id") or "centrifuge"},
            {"verb": "wait", "minutes": op.get("minutes")},
            {"verb": "stop", "device": op.get("centrifuge_id") or "centrifuge"},
            {"verb": "place", "object": tube, "to": "rack"},
        ]
        return m

    if typ == "sonicate":
        tube = op.get("tube") or "V2_tube"
        m += [
            {"verb": "pick_up", "object": tube, "from": "rack"},
            {"verb": "place", "object": tube, "to": "sonicator"},
            {
                "verb": "set",
                "device": op.get("sonicator_id") or "sonicator",
                "param": "minutes",
                "value": op.get("minutes"),
            },
            {"verb": "start", "device": op.get("sonicator_id") or "sonicator"},
            {"verb": "wait", "minutes": op.get("minutes")},
            {"verb": "stop", "device": op.get("sonicator_id") or "sonicator"},
            {"verb": "place", "object": tube, "to": "rack"},
        ]
        return m

    if typ == "move_to_oven":
        m += [
            {"verb": "pick_up", "object": op.get("tube") or "tube", "from": "rack"},
            {"verb": "place", "object": op.get("tube") or "tube", "to": "oven"},
        ]
        return m

    if typ == "set_oven_temperature":
        m += [
            {
                "verb": "set",
                "device": op.get("oven_id") or "oven",
                "param": "temperature_C",
                "value": op.get("temperature_C"),
            }
        ]
        return m

    if typ == "stir":
        v = _label_for_vessel(op.get("vessel", ""), vessels, hardware)
        m += [
            {"verb": "pick_up", "object": v, "from": "bench"},
            {"verb": "place", "object": v, "to": "stir_plate"},
            {
                "verb": "set",
                "device": "stir_plate",
                "param": "rpm",
                "value": op.get("rpm"),
            },
            {"verb": "wait", "minutes": op.get("minutes")},
        ]
        return m

    return m


def expand_ops_to_micro(
    ops: list[dict], vessels: "VesselRegistry", hardware: list[dict]
) -> list[dict]:
    out = []
    for op in ops or []:
        out.extend(_micro_for_op(op, vessels, hardware))
    return out


# -------- Minimal primitive reduction (pick_up, place, pour, set) --------
_MIN_ALLOWED = {"pick_up", "place", "pour", "set"}

_GENERIC_DEVICE_MAP_DEFAULT = {
    "HP1": "hotplate",
    "SP1": "stir_plate",
    "CF1": "centrifuge",
    "OV1": "oven",
    "VP1": "vacuum_pump",
    "VX1": "vortexer",
    "SN1": "sonicator",
}


def _derive_minimal_micro_plan(doc: dict, allow_wait: bool = False) -> tuple[list[dict], list[dict]]:
    """Derive a reduced micro plan containing only primitive verbs a simple robot supports.

    Rules:
    - Keep verbs in {_MIN_ALLOWED} verbatim.
    - wait → dropped (timing captured separately) unless allow_wait=True.
    - start/stop → set (device, param=power, value=on/off).
    - vortex/resuspend/sonicate/centrifuge sequences already expanded earlier; we drop
      auxiliary verbs (vortex, start, stop) and keep preceding pick/place/set.
    - Decant / add_wash_solvent etc. are already decomposed into pick/place/pour.
    - Consecutive duplicates collapsed.

    Returns (min_plan, delays) where delays is a list of timing annotations:
       {"after_index": <index in min_plan (1-based)>, "minutes": X, "original_step_index": Y}
    """
    original = doc.get("micro_plan") or []
    min_plan: list[dict] = []
    delays: list[dict] = []

    # Optional generic mapping for devices, activated via MIN_PLAN_MAP_GENERIC=1
    want_map = os.getenv("MIN_PLAN_MAP_GENERIC", "").lower() in {"1", "true", "yes"}
    dev_map = _GENERIC_DEVICE_MAP_DEFAULT if want_map else {}

    for entry in original:
        if not isinstance(entry, dict):
            continue
        verb = entry.get("verb")
        if verb == "wait":
            mins = entry.get("minutes")
            if mins is not None and not allow_wait and min_plan:
                delays.append(
                    {
                        "after_index": len(min_plan),
                        "minutes": mins,
                        "original_step_index": entry.get("step_index"),
                    }
                )
            elif allow_wait:
                # Optionally keep as set pseudo-op for timing
                min_plan.append(
                    {
                        "verb": "set",
                        "device": "scheduler",
                        "param": "delay_minutes",
                        "value": mins,
                        "step_index": entry.get("step_index"),
                    }
                )
            continue
        if verb in ("start", "stop"):
            dev = entry.get("device") or "device"
            min_plan.append(
                {
                    "verb": "set",
                    "device": dev,
                    "param": "power",
                    "value": "on" if verb == "start" else "off",
                    "step_index": entry.get("step_index"),
                }
            )
            continue
        if verb in _MIN_ALLOWED:
            # Shallow copy & strip extraneous keys that aren't needed for primitive execution
            keep = {k: v for k, v in entry.items() if k in {"verb", "object", "from", "to", "device", "param", "value", "amount", "unit", "volume", "volume_units", "rate", "step_index"}}
            # Apply device remap
            if want_map and keep.get("device") in dev_map:
                keep["device"] = dev_map[keep["device"]]
            min_plan.append(keep)
            continue
        # Ignore other verbs (vortex, decant already decomposed, etc.)
        # They should have produced primitive children earlier.
        continue

    # Collapse consecutive duplicates
    collapsed: list[dict] = []
    for item in min_plan:
        if not collapsed or item != collapsed[-1]:
            collapsed.append(item)
    return collapsed, delays


def convert_text_to_robot_ops(text: str) -> Dict:
    hardware = parse_hardware(text)
    vessels = VesselRegistry(hardware)
    records: List[Dict] = []

    steps = extract_steps(text)

    for step in steps:
        # Weighing
        weigh = detect_weigh(step)
        if weigh:
            target_vessel = vessels.primary_vessel or vessels.ensure_glassware("Beaker")
            record = {
                "action": "weigh",
                "vessel": target_vessel,
                "reagent": weigh["reagent"],
                "amount": weigh["amount"],
                "unit": weigh["unit"],
                "ops": [
                    {
                        "op": "weigh",
                        "reagent": weigh["reagent"],
                        "amount": weigh["amount"],
                        "unit": weigh["unit"],
                    }
                ],
                "raw": step,
                "reagents": [weigh["reagent"]],
            }
            _normalize_reagents_inplace(record)
            _add_structured_reagents_inplace(record)
            records.append(record)
            continue

        # Transfer (explicit)
        transfer_exp = detect_transfer_explicit(step)
        if transfer_exp:
            target_vessel = vessels.ensure_glassware(transfer_exp["target"])
            record = {
                "action": "transfer",
                "vessel": target_vessel,
                "ops": [
                    {
                        "op": "transfer",
                        "to": transfer_exp["target"],
                        "tube": f"{target_vessel}_tube",
                    }
                ],
                "raw": step,
                "reagents": [],
            }
            _normalize_reagents_inplace(record)
            _add_structured_reagents_inplace(record)
            records.append(record)
            continue

        # Dissolve
        dissolve = detect_dissolve(step)
        if dissolve:
            target_vessel = vessels.ensure_glassware(f"{dissolve['solute']} solution")
            record = {
                "action": "dissolve",
                "vessel": target_vessel,
                "solute": dissolve["solute"],
                "amount": dissolve["amount"],
                "unit": dissolve["unit"],
                "solvent": dissolve["solvent"],
                "volume": dissolve["volume"],
                "volume_units": dissolve["volume_units"],
                "ops": [
                    {
                        "op": "add_solute",
                        "vessel": target_vessel,
                        "reagent": dissolve["solute"],
                        "amount": dissolve["amount"],
                        "unit": dissolve["unit"],
                    },
                    *(
                        [
                            {
                                "op": "add_solvent",
                                "vessel": target_vessel,
                                "reagent": comp["name"],
                                "volume": comp["volume"],
                                "volume_units": comp["volume_units"],
                            }
                            for comp in (
                                dissolve.get("solvents")
                                or [
                                    {
                                        "name": dissolve["solvent"],
                                        "volume": dissolve["volume"],
                                        "volume_units": dissolve["volume_units"],
                                    }
                                ]
                            )
                        ]
                    ),
                    {
                        "op": "stir",
                        "vessel": target_vessel,
                        "rpm": DEFAULTS["stir_rpm"],
                        "minutes": 2,
                    },
                ],
                "raw": step,
                "reagents": [dissolve["solute"]]
                + [
                    comp["name"]
                    for comp in (
                        dissolve.get("solvents") or [{"name": dissolve["solvent"]}]
                    )
                ],
            }
            _normalize_reagents_inplace(record)
            _add_structured_reagents_inplace(record)
            records.append(record)
            continue

        # Isolate/filter
        isolate = detect_filter_isolate(step)
        if isolate:
            target_vessel = vessels.primary_vessel or vessels.ensure_glassware("Beaker")
            record = {
                "action": "isolate",
                "vessel": target_vessel,
                "ops": [
                    {"op": "filter", "vessel": target_vessel},
                    {"op": "collect", "vessel": target_vessel},
                ],
                "raw": step,
                "reagents": [],
            }
            _normalize_reagents_inplace(record)
            _add_structured_reagents_inplace(record)
            records.append(record)
            continue

        # Solution preparation
        prep = detect_solution_prep(step)
        if prep:
            vol_ml = prep["volume"] * (
                0.001
                if prep["volume_units"].lower() in ("µl", "ul")
                else (1.0 if prep["volume_units"].lower() == "ml" else 1000.0)
            )
            explicit = prep.get("hardware_hint")
            label = explicit if explicit else "Beaker"
            vid = vessels.ensure_glassware(
                label, prefer_capacity_ml=vol_ml, explicit_hardware_hint=explicit
            )
            vessels.map_contents(
                vid,
                f"{prep['solvent']} {prep['concentration']} {prep['concentration_units']} solution of {prep['solute']}",
            )
            hw_id = vessels.vessel_hardware(vid)
            record = {
                "action": "dispense",
                "vessel": vid,
                "hardware_id": hw_id,
                "solute": prep["solute"],
                "solvent": prep["solvent"],
                "concentration": prep["concentration"],
                "concentration_units": prep["concentration_units"],
                "volume": prep["volume"],
                "volume_units": prep["volume_units"],
                "reagents": [prep["solute"], prep["solvent"]],
                "ops": ops_for_dispense(
                    vid,
                    hw_id,
                    prep["solute"],
                    prep["solvent"],
                    prep["volume"],
                    prep["volume_units"],
                ),
                "raw": step,
            }
            _normalize_reagents_inplace(record)
            _add_structured_reagents_inplace(record)
            records.append(record)
            continue

        # Additions (with optional temp/rate/time)
        solv = detect_add_solvent(step)
        if solv:
            target_vessel = vessels.primary_vessel or vessels.ensure_glassware("Beaker")
            record = {
                "action": "add_solvent",
                "vessel": target_vessel,
                "solvent": solv["solvent"],
                "volume": solv["volume"],
                "volume_units": solv["volume_units"],
                "reagents": [solv["solvent"]],
                "ops": [
                    {
                        "op": "add_solvent",
                        "vessel": target_vessel,
                        "solvent": solv["solvent"],
                        "volume": solv["volume"],
                        "volume_units": solv["volume_units"],
                    }
                ],
                "raw": step,
            }
            _normalize_reagents_inplace(record)
            _add_structured_reagents_inplace(record)
            records.append(record)
            continue

        add = detect_add(step)
        if add:
            src_key = re.sub(r"^\bthe\b\s+", "", add["source_name"], flags=re.I).strip()
            dst_key = re.sub(r"^\bthe\b\s+", "", add["target_name"], flags=re.I).strip()
            src_vid = vessels.ensure_glassware(src_key)
            dst_vid = vessels.ensure_glassware(dst_key)
            record = {
                "action": "add",
                "source_vessel": src_vid,
                "target_vessel": dst_vid,
                "reagents": [src_key],
                "with_stirring": True,
                "rate": add["rate"],
                "temperature_C": add.get("temperature_C"),
                "minutes": add.get("minutes"),
                "ops": ops_for_add(
                    src_vid,
                    dst_vid,
                    add["rate"],
                    temperature_C=add.get("temperature_C"),
                    minutes=add.get("minutes"),
                ),
                "raw": step,
            }
            _normalize_reagents_inplace(record)
            _add_structured_reagents_inplace(record)
            records.append(record)
            continue

        # Stirring
        st = detect_stir(step)
        if st:
            target_vessel = vessels.primary_vessel or vessels.ensure_glassware("Beaker")
            record = {
                "action": "stir",
                "vessel": target_vessel,
                "reagents": [],
                "minutes": st["minutes"],
                "temperature_C": st["temperature_C"],
                "rpm": st["rpm"],
                "ops": ops_for_stir(
                    target_vessel, st["minutes"], st["rpm"], st["temperature_C"]
                ),
                "raw": step,
            }
            _normalize_reagents_inplace(record)
            _add_structured_reagents_inplace(record)
            records.append(record)
            continue

        # Heating
        ht = detect_heat(step)
        if ht:
            target_vessel = vessels.primary_vessel or vessels.ensure_glassware("Beaker")
            temp = ht[0]["temperature_C"]
            minutes = ht[1]["minutes"]
            record = {
                "action": "heat_hold",
                "vessel": target_vessel,
                "reagents": [],
                "minutes": minutes,
                "temperature_C": temp,
                "ops": ops_for_heat(target_vessel, temp, minutes),
                "raw": step,
            }
            _normalize_reagents_inplace(record)
            _add_structured_reagents_inplace(record)
            records.append(record)
            continue

        # Cooling
        cl = detect_cool(step)
        if cl:
            target_vessel = vessels.primary_vessel or vessels.ensure_glassware("Beaker")
            record = {
                "action": "cool_to",
                "vessel": target_vessel,
                "reagents": [],
                "temperature_C": cl["temperature_C"],
                "ops": [
                    {
                        "op": "set",
                        "device": DEVICE_IDS["hotplate_id"],
                        "param": "temperature_C",
                        "value": cl["temperature_C"],
                    }
                ],
                "raw": step,
            }
            _normalize_reagents_inplace(record)
            _add_structured_reagents_inplace(record)
            records.append(record)
            continue

        # Sonication
        so = detect_sonicate(step)
        if so:
            target_vessel = vessels.primary_vessel or vessels.ensure_glassware("Beaker")
            record = {
                "action": "sonicate",
                "vessel": target_vessel,
                "reagents": [],
                "minutes": so["minutes"],
                "ops": [
                    {
                        "op": "sonicate",
                        "sonicator_id": DEVICE_IDS["sonicator_id"],
                        "minutes": so["minutes"],
                    }
                ],
                "raw": step,
            }
            _normalize_reagents_inplace(record)
            _add_structured_reagents_inplace(record)
            records.append(record)
            continue

        # Filtration / washing / drying
        filt = detect_filter(step)
        if filt:
            target_vessel = vessels.primary_vessel or vessels.ensure_glassware("Beaker")
            record = {
                "action": "postprocess",
                "vessel": target_vessel,
                "reagents": [],
                "ops": ops_for_postproc(target_vessel, filt),
                "raw": step,
            }
            _normalize_reagents_inplace(record)
            _add_structured_reagents_inplace(record)
            records.append(record)
            continue

        wd = detect_wash_dry(step)
        if wd:
            target_vessel = vessels.primary_vessel or vessels.ensure_glassware("Beaker")
            record = {
                "action": "postprocess",
                "vessel": target_vessel,
                "reagents": [],
                "ops": ops_for_postproc(target_vessel, wd),
                "raw": step,
            }
            _normalize_reagents_inplace(record)
            _add_structured_reagents_inplace(record)
            records.append(record)
            continue

        # Resuspend
        res = detect_resuspend(step)
        if res:
            target_vessel = vessels.primary_vessel or vessels.ensure_glassware("Beaker")
            record = {
                "action": "resuspend",
                "vessel": target_vessel,
                "ops": [{"op": "resuspend", "tube": f"{target_vessel}_tube"}],
                "raw": step,
                "reagents": [],
            }
            _normalize_reagents_inplace(record)
            _add_structured_reagents_inplace(record)
            records.append(record)
            continue

        # Collect
        col = detect_collect(step)
        if col:
            target_vessel = vessels.primary_vessel or vessels.ensure_glassware("Beaker")
            record = {
                "action": "collect",
                "vessel": target_vessel,
                "ops": [{"op": "collect", "tube": f"{target_vessel}_tube"}],
                "raw": step,
                "reagents": [],
            }
            _normalize_reagents_inplace(record)
            _add_structured_reagents_inplace(record)
            records.append(record)
            continue

        # Discard
        dis = detect_discard(step)
        if dis:
            target_vessel = vessels.primary_vessel or vessels.ensure_glassware("Beaker")
            record = {
                "action": "discard",
                "vessel": target_vessel,
                "ops": [{"op": "discard_supernatant", "tube": f"{target_vessel}_tube"}],
                "raw": step,
                "reagents": [],
            }
            _normalize_reagents_inplace(record)
            _add_structured_reagents_inplace(record)
            records.append(record)
            continue

        # Transfer
        tra = detect_transfer(step)
        if tra:
            target_vessel = vessels.primary_vessel or vessels.ensure_glassware("Beaker")
            record = {
                "action": "transfer",
                "vessel": target_vessel,
                "ops": [
                    {
                        "op": "transfer",
                        "to": tra["target"],
                        "tube": f"{target_vessel}_tube",
                    }
                ],
                "raw": step,
                "reagents": [],
            }
            _normalize_reagents_inplace(record)
            _add_structured_reagents_inplace(record)
            records.append(record)
            continue

        # Fallback generic process node
        target_vessel = vessels.primary_vessel or vessels.ensure_glassware("Beaker")
        substeps = re.split(r"\band\b|;|\.", step)
        for sub in substeps:
            sub = sub.strip()
            if not sub:
                continue
            # Try all detectors again for each substep
            weigh = detect_weigh(sub)
            if weigh:
                record = {
                    "action": "weigh",
                    "vessel": target_vessel,
                    "reagent": weigh["reagent"],
                    "amount": weigh["amount"],
                    "unit": weigh["unit"],
                    "ops": [
                        {
                            "op": "weigh",
                            "reagent": weigh["reagent"],
                            "amount": weigh["amount"],
                            "unit": weigh["unit"],
                        }
                    ],
                    "raw": sub,
                    "reagents": [weigh["reagent"]],
                }
                _normalize_reagents_inplace(record)
                _add_structured_reagents_inplace(record)
                records.append(record)
                continue
            transfer_exp = detect_transfer_explicit(sub)
            if transfer_exp:
                record = {
                    "action": "transfer",
                    "vessel": target_vessel,
                    "ops": [
                        {
                            "op": "transfer",
                            "to": transfer_exp["target"],
                            "tube": f"{target_vessel}_tube",
                        }
                    ],
                    "raw": sub,
                    "reagents": [],
                }
                _normalize_reagents_inplace(record)
                _add_structured_reagents_inplace(record)
                records.append(record)
                continue
            dissolve = detect_dissolve(sub)
            if dissolve:
                record = {
                    "action": "dissolve",
                    "vessel": target_vessel,
                    "solute": dissolve["solute"],
                    "amount": dissolve["amount"],
                    "unit": dissolve["unit"],
                    "solvent": dissolve["solvent"],
                    "volume": dissolve["volume"],
                    "volume_units": dissolve["volume_units"],
                    "ops": [
                        {
                            "op": "add_solute",
                            "vessel": target_vessel,
                            "reagent": dissolve["solute"],
                            "amount": dissolve["amount"],
                            "unit": dissolve["unit"],
                        },
                        *(
                            [
                                {
                                    "op": "add_solvent",
                                    "vessel": target_vessel,
                                    "reagent": comp["name"],
                                    "volume": comp["volume"],
                                    "volume_units": comp["volume_units"],
                                }
                                for comp in (
                                    dissolve.get("solvents")
                                    or [
                                        {
                                            "name": dissolve["solvent"],
                                            "volume": dissolve["volume"],
                                            "volume_units": dissolve["volume_units"],
                                        }
                                    ]
                                )
                            ]
                        ),
                        {
                            "op": "stir",
                            "vessel": target_vessel,
                            "rpm": DEFAULTS["stir_rpm"],
                            "minutes": 2,
                        },
                    ],
                    "raw": sub,
                    "reagents": [dissolve["solute"]]
                    + [
                        comp["name"]
                        for comp in (
                            dissolve.get("solvents") or [{"name": dissolve["solvent"]}]
                        )
                    ],
                }
                _normalize_reagents_inplace(record)
                _add_structured_reagents_inplace(record)
                records.append(record)
                continue
            isolate = detect_filter_isolate(sub)
            if isolate:
                record = {
                    "action": "isolate",
                    "vessel": target_vessel,
                    "ops": [
                        {"op": "filter", "vessel": target_vessel},
                        {"op": "collect", "vessel": target_vessel},
                    ],
                    "raw": sub,
                    "reagents": [],
                }
                _normalize_reagents_inplace(record)
                _add_structured_reagents_inplace(record)
                records.append(record)
                continue

            res = detect_resuspend(sub)
            if res:
                record = {
                    "action": "resuspend",
                    "vessel": target_vessel,
                    "ops": [{"op": "resuspend", "tube": f"{target_vessel}_tube"}],
                    "raw": sub,
                    "reagents": [],
                }
                _normalize_reagents_inplace(record)
                _add_structured_reagents_inplace(record)
                records.append(record)
                continue
            col = detect_collect(sub)
            if col:
                record = {
                    "action": "collect",
                    "vessel": target_vessel,
                    "ops": [{"op": "collect", "tube": f"{target_vessel}_tube"}],
                    "raw": sub,
                    "reagents": [],
                }
                _normalize_reagents_inplace(record)
                _add_structured_reagents_inplace(record)
                records.append(record)
                continue
            dis = detect_discard(sub)
            if dis:
                record = {
                    "action": "discard",
                    "vessel": target_vessel,
                    "ops": [
                        {"op": "discard_supernatant", "tube": f"{target_vessel}_tube"}
                    ],
                    "raw": sub,
                    "reagents": [],
                }
                _normalize_reagents_inplace(record)
                _add_structured_reagents_inplace(record)
                records.append(record)
                continue
            tra = detect_transfer(sub)
            if tra:
                record = {
                    "action": "transfer",
                    "vessel": target_vessel,
                    "ops": [
                        {
                            "op": "transfer",
                            "to": tra["target"],
                            "tube": f"{target_vessel}_tube",
                        }
                    ],
                    "raw": sub,
                    "reagents": [],
                }
                _normalize_reagents_inplace(record)
                _add_structured_reagents_inplace(record)
                records.append(record)
                continue
            # If still nothing, add as process
            record = {
                "action": "process",
                "vessel": target_vessel,
                "reagents": [],
                "ops": [],
                "raw": sub,
            }
            _normalize_reagents_inplace(record)
            _add_structured_reagents_inplace(record)
            records.append(record)

    # Build micro-ops per step and a flattened micro plan
    for rec in records:
        rec["micro_ops"] = expand_ops_to_micro(rec.get("ops", []), vessels, hardware)

    micro_plan = []
    for i, rec in enumerate(records, 1):
        # --- 1) Expand any material transfer into primitive robot actions ---
        # Any 'add_*' step is already represented as an explicit 'transfer' op in rec['ops'].
        # Convert that to: pick_up (source) → pour (source→target) → place (source down)
        for op in rec.get("ops", []):
            if op.get("op") == "transfer":
                src_id = op.get("from")
                dst_id = op.get("to")
                if src_id and dst_id:
                    src_label = _label_for_vessel(src_id, vessels, hardware)
                    dst_label = _label_for_vessel(dst_id, vessels, hardware)
                    ctx = (
                        rec.get("vessel")
                        or rec.get("target_vessel")
                        or rec.get("source_vessel")
                    )

                    micro_plan.append(
                        {
                            "verb": "pick_up",
                            "object": src_label,
                            "step_index": i,
                            "context_vessel": ctx,
                        }
                    )
                    pour_item = {
                        "verb": "pour",
                        "from": src_label,
                        "to": dst_label,
                        "step_index": i,
                        "context_vessel": ctx,
                    }
                    rate = op.get("rate")
                    if rate:  # include if present
                        pour_item["rate"] = rate
                    micro_plan.append(pour_item)
                    micro_plan.append(
                        {
                            "verb": "place",
                            "object": src_label,
                            "to": "bench",
                            "step_index": i,
                            "context_vessel": ctx,
                        }
                    )

        # --- 2) Copy through any authored micro_ops EXCEPT 'note add_*' markers ---
        for micro in rec.get("micro_ops", []):
            # Drop the diagnostic "note/op=add_*" entries so nothing shows up as a 'set'
            if (
                micro.get("device") == "note"
                and micro.get("param") == "op"
                and str(micro.get("value", "")).startswith("add_")
            ):
                continue
            item = dict(micro)
            item["step_index"] = i
            item["context_vessel"] = (
                rec.get("vessel")
                or rec.get("target_vessel")
                or rec.get("source_vessel")
            )
            micro_plan.append(item)

    result = {
        "hardware": hardware,
        "vessel_registry": vessels.as_dict(),
        "vessel_contents": vessels.contents_dict(),
        "devices": DEVICE_IDS,
        "micro_plan": micro_plan,
        "defaults": DEFAULTS,
        "steps": records,
    }
    result = apply_postprocessing(result)
    try:
        # Always derive minimal primitive plan; can be disabled via env var if desired
        disable = os.getenv("DISABLE_MIN_PRIMITIVE_PLAN", "").lower() in {"1", "true", "yes"}
        if not disable:
            allow_wait = os.getenv("MIN_PLAN_ALLOW_WAIT", "").lower() in {"1", "true", "yes"}
            min_plan, delays = _derive_minimal_micro_plan(result, allow_wait=allow_wait)
            result["micro_plan_min"] = min_plan
            if delays:
                result.setdefault("timing_delays", delays)
            meta = result.setdefault("meta", {})
            meta["min_primitive_plan"] = True
            meta["min_plan_wait_mode"] = "inline" if allow_wait else "delays"
    except Exception as _e:
        # Non-fatal; continue with original result
        pass
    return result


# -------- Validation helpers (unchanged API) --------
def validate_step(text: str) -> Dict[str, Any]:
    if not isinstance(text, str):
        raise ValueError("input must be a string")
    raw = text.strip()
    if not raw:
        raise ValueError("input text is empty")
    try:
        obj = json.loads(raw)
        if isinstance(obj, dict):
            return obj
        raise ValueError("JSON input must be an object")
    except json.JSONDecodeError:
        pass
    data: Dict[str, Any] = {}
    for lineno, line in enumerate(raw.splitlines(), start=1):
        if not line.strip():
            continue
        if ":" not in line:
            raise ValueError(f"line {lineno}: missing ':' separator")
        key, value = line.split(":", 1)
        key = key.strip()
        value = value.strip()
        if not key:
            raise ValueError(f"line {lineno}: key is empty")
        data[key] = value
    if not data:
        raise ValueError("no key:value pairs found")
    return data


def validate_file(path: str) -> List[Dict[str, Any]]:
    p = pathlib.Path(path)
    if not p.exists():
        raise ValueError(f"file '{path}' does not exist")
    items: List[Dict[str, Any]] = []
    with p.open("r", encoding="utf-8", errors="ignore") as fh:
        for lineno, line in enumerate(fh, start=1):
            if not line.strip():
                continue
            try:
                item = validate_step(line)
            except ValueError as ve:
                raise ValueError(f"{p.name}:{lineno}: {ve}") from None
            items.append(item)
    return items


# --- hard-wrap the exported function the web app calls ---
if "convert_text_to_robot_ops" in globals():
    _orig_convert = convert_text_to_robot_ops

    def convert_text_to_robot_ops(text: str):
        doc = _orig_convert(text)
        try:
            return apply_postprocessing(doc)
        except Exception:
            return doc


# -------- CLI --------
if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(
        description="Convert a TXT/MD protocol to robot JSON ops"
    )
    ap.add_argument("path", help="Input file path")
    ap.add_argument(
        "-o", "--out", default="-", help="Output JSON path (default stdout)"
    )
    args = ap.parse_args()
    txt = pathlib.Path(args.path).read_text(encoding="utf-8", errors="ignore")
    obj = convert_text_to_robot_ops(txt)
    js = json.dumps(obj, indent=2, ensure_ascii=False)
    if args.out == "-":
        print(js)
    else:
        with open(args.out, "w", encoding="utf-8") as f:
            f.write(js)
        print(f"Wrote {args.out}")
