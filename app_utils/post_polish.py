"""
post_polish.py — generalized post-processing for lab automation JSON.

Usage
-----
As a library:
    from post_polish import polish_robot_doc
    cleaned = polish_robot_doc(doc, config={
        "reaction_vessel": "V1",
        "bottle_map": {
            "oleic acid": "oleic_bottle",
            "1-octadecene": "ode_bottle",
            "ethanol": "ethanol_bottle",
            "acetone": "acetone_bottle",
            "solute": "solute_container",
            "deionized water": "water_bottle"
        },
        "bottle_labels": {
            "oleic_bottle": "Oleic acid bottle",
            "ode_bottle": "1-Octadecene bottle",
            "ethanol_bottle": "Ethanol bottle",
            "acetone_bottle": "Acetone bottle",
            "solute_container": "Generic reagent container",
            "water_bottle": "Deionized water bottle",
            "waste": "Waste container"
        },
        "devices": {"hotplate_id":"HP1","stir_plate_id":"SP1","centrifuge_id":"CF1","vacuum_pump_id":"VP1","oven_id":"OV1"},
        "centrifuge": {"rpm": 4000, "minutes": 10, "tube": "V2_tube"},
        "wash": {"reagent":"ethanol_bottle","cycles": 0},
        "drying": {"prefer_ambient_if_mentioned": True, "ambient_minutes": 1440, "vacuum_minutes": 720, "vacuum_temp_C": 25}
    })

CLI:
    python post_polish.py input.json output.json

Notes
-----
- Only allowed micro verbs are preserved: pick_up, place, pour, set, wait.
- start/stop → set(power=on/off), decant → pour(...→ waste).
- micro_plan is rebuilt from steps[*].micro_ops to ensure they match.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List

ALLOWED = {"pick_up", "place", "pour", "set", "wait"}

DEFAULT_CONFIG: Dict[str, Any] = {
    "reaction_vessel": "V1",
    "bottle_map": {},
    "bottle_labels": {"waste": "Waste container"},
    "devices": {
        "hotplate_id": "HP1",
        "stir_plate_id": "SP1",
        "centrifuge_id": "CF1",
        "vacuum_pump_id": "VP1",
        "oven_id": "OV1",
    },
    "centrifuge": {"rpm": 4000, "minutes": 10, "tube": "V2_tube"},
    "wash": {"reagent": None, "cycles": 0},  # e.g., "ethanol_bottle", cycles=2
    "drying": {
        "prefer_ambient_if_mentioned": True,
        "ambient_minutes": 1440,
        "vacuum_minutes": 720,
        "vacuum_temp_C": 25,
    },
}


def _deep_get(d: Dict[str, Any], *path, default=None):
    cur = d
    for p in path:
        if not isinstance(cur, dict) or p not in cur:
            return default
        cur = cur[p]
    return cur


def _clean_reagent_names(doc: Dict[str, Any]) -> None:
    for st in doc.get("steps", []) or []:
        for r in st.get("reagents_structured") or []:
            name = r.get("name")
            if isinstance(name, str):
                name2 = re.sub(r"\.\s*Heat the mixture.*$", "", name, flags=re.I)
                name2 = re.sub(r"^\s*of\s+", "", name2, flags=re.I)
                r["name"] = name2.strip()


def _map_materials_to_vessels(doc: Dict[str, Any], cfg: Dict[str, Any]) -> None:
    bottle_map = cfg.get("bottle_map", {}) or {}
    # pass 1: rewrite in steps
    for st in doc.get("steps", []) or []:
        ops = st.get("micro_ops") or []
        new = []
        for m in ops:
            m = dict(m)
            if m.get("verb") in {"pick_up", "place"} and isinstance(
                m.get("object"), str
            ):
                m["object"] = bottle_map.get(
                    m["object"].lower(), bottle_map.get(m["object"], m["object"])
                )
            if m.get("verb") == "pour" and isinstance(m.get("from"), str):
                m["from"] = bottle_map.get(
                    m["from"].lower(), bottle_map.get(m["from"], m["from"])
                )
            new.append(m)
        st["micro_ops"] = new


def _normalize_heating_placement(doc: Dict[str, Any], cfg: Dict[str, Any]) -> None:
    hp = _deep_get(cfg, "devices", "hotplate_id", default="HP1")
    for st in doc.get("steps", []) or []:
        mops = st.get("micro_ops") or []
        if any(
            m.get("verb") == "set"
            and m.get("device") == hp
            and m.get("param") == "temperature_C"
            for m in mops
        ):
            for m in mops:
                if (
                    m.get("verb") == "place"
                    and isinstance(m.get("object"), str)
                    and re.fullmatch(r"V\d+(_tube)?", m["object"])
                ):
                    m["to"] = hp


def _canonicalize_centrifuge(doc: Dict[str, Any], cfg: Dict[str, Any]) -> None:
    cf_id = _deep_get(cfg, "devices", "centrifuge_id", default="CF1")
    tube = _deep_get(cfg, "centrifuge", "tube", default="V2_tube")
    rpm = _deep_get(cfg, "centrifuge", "rpm", default=4000)
    mins_default = _deep_get(cfg, "centrifuge", "minutes", default=10)

    for st in doc.get("steps", []) or []:
        raw = (st.get("raw", "") or "").lower()
        if "centrifuge" not in raw:
            continue
        idx = st.get("index") or 1
        mins = st.get("minutes", mins_default) or mins_default
        # always rebuild sequence to be safe
        st["micro_ops"] = [
            {
                "verb": "pour",
                "from": cfg.get("reaction_vessel", "V1"),
                "to": tube,
                "step_index": idx,
            },
            {"verb": "pick_up", "object": tube, "from": "rack", "step_index": idx},
            {"verb": "place", "object": tube, "to": cf_id, "step_index": idx},
            {
                "verb": "set",
                "device": cf_id,
                "param": "rpm",
                "value": rpm,
                "step_index": idx,
            },
            {
                "verb": "set",
                "device": cf_id,
                "param": "power",
                "value": "on",
                "step_index": idx,
            },
            {"verb": "wait", "minutes": mins, "step_index": idx},
            {
                "verb": "set",
                "device": cf_id,
                "param": "power",
                "value": "off",
                "step_index": idx,
            },
            {"verb": "place", "object": tube, "to": "rack", "step_index": idx},
            {"verb": "pour", "from": tube, "to": "waste", "step_index": idx},
        ]

        # Optional wash cycles appended after a centrifuge step if "wash" in raw or wash.cycles>0
        wash_cfg = cfg.get("wash", {}) or {}
        cycles = wash_cfg.get("cycles", 0) or 0
        reagent = wash_cfg.get("reagent")  # e.g., "ethanol_bottle"
        if ("wash" in raw or cycles > 0) and reagent:
            for _ in range(cycles):
                st["micro_ops"] += [
                    {"verb": "pour", "from": reagent, "to": tube, "step_index": idx},
                    {
                        "verb": "pick_up",
                        "object": tube,
                        "from": "rack",
                        "step_index": idx,
                    },
                    {"verb": "place", "object": tube, "to": cf_id, "step_index": idx},
                    {
                        "verb": "set",
                        "device": cf_id,
                        "param": "rpm",
                        "value": rpm,
                        "step_index": idx,
                    },
                    {
                        "verb": "set",
                        "device": cf_id,
                        "param": "power",
                        "value": "on",
                        "step_index": idx,
                    },
                    {"verb": "wait", "minutes": mins, "step_index": idx},
                    {
                        "verb": "set",
                        "device": cf_id,
                        "param": "power",
                        "value": "off",
                        "step_index": idx,
                    },
                    {"verb": "place", "object": tube, "to": "rack", "step_index": idx},
                    {"verb": "pour", "from": tube, "to": "waste", "step_index": idx},
                ]


def _enforce_drying(doc: Dict[str, Any], cfg: Dict[str, Any]) -> None:
    vp = _deep_get(cfg, "devices", "vacuum_pump_id", default="VP1")
    ov = _deep_get(cfg, "devices", "oven_id", default="OV1")
    pref_amb = _deep_get(cfg, "drying", "prefer_ambient_if_mentioned", default=True)
    amb_min = _deep_get(cfg, "drying", "ambient_minutes", default=1440)
    vac_min = _deep_get(cfg, "drying", "vacuum_minutes", default=720)
    vac_T = _deep_get(cfg, "drying", "vacuum_temp_C", default=25)

    for st in doc.get("steps", []) or []:
        raw = (st.get("raw", "") or "").lower()
        idx = st.get("index") or 1
        mentions_amb = "ambient" in raw
        mentions_vac = "vacuum" in raw
        mentions_dry = "dry" in raw

        if not mentions_dry:
            continue

        if mentions_amb and pref_amb:
            # Ambient rule wins
            st["minutes"] = st.get("minutes") or amb_min
            st["micro_ops"] = [
                {
                    "verb": "place",
                    "object": _deep_get(cfg, "centrifuge", "tube", default="V2_tube"),
                    "to": "bench",
                    "step_index": idx,
                },
                {"verb": "wait", "minutes": st["minutes"], "step_index": idx},
            ]
        elif mentions_vac:
            # Vacuum rule
            mins = st.get("minutes") or vac_min
            st["minutes"] = mins
            st["micro_ops"] = [
                {
                    "verb": "place",
                    "object": _deep_get(cfg, "centrifuge", "tube", default="V2_tube"),
                    "to": ov,
                    "step_index": idx,
                },
                {
                    "verb": "set",
                    "device": ov,
                    "param": "temperature_C",
                    "value": vac_T,
                    "step_index": idx,
                },
                {
                    "verb": "set",
                    "device": vp,
                    "param": "power",
                    "value": "on",
                    "step_index": idx,
                },
                {"verb": "wait", "minutes": mins, "step_index": idx},
                {
                    "verb": "set",
                    "device": vp,
                    "param": "power",
                    "value": "off",
                    "step_index": idx,
                },
            ]


def _ensure_registry(doc: Dict[str, Any], cfg: Dict[str, Any]) -> None:
    reg = doc.setdefault("vessel_registry", {}) or {}
    # add bottle labels
    for vid, label in (cfg.get("bottle_labels") or {}).items():
        reg.setdefault(vid, label)

    # auto-register any referenced V* in steps
    def maybe_add(v):
        if isinstance(v, str) and (
            v.startswith("V") or v.endswith("_bottle") or v in ("waste", "bench")
        ):
            reg.setdefault(v, "(auto) vessel")

    for st in doc.get("steps", []) or []:
        for m in st.get("micro_ops") or []:
            for k in ("object", "from", "to"):
                maybe_add(m.get(k))
    doc["vessel_registry"] = reg


def _enforce_base_verbs(doc: Dict[str, Any]) -> None:
    # Map start/stop/decant → set/pour, drop unknown verbs
    for st in doc.get("steps", []) or []:
        new = []
        for m in st.get("micro_ops") or []:
            v = m.get("verb")
            if v == "start":
                dev = m.get("device") or "device"
                m = {
                    "verb": "set",
                    "device": dev,
                    "param": "power",
                    "value": "on",
                    "step_index": m.get("step_index"),
                }
            elif v == "stop":
                dev = m.get("device") or "device"
                m = {
                    "verb": "set",
                    "device": dev,
                    "param": "power",
                    "value": "off",
                    "step_index": m.get("step_index"),
                }
            elif v == "decant":
                src = m.get("object") or m.get("from") or "tube"
                m = {
                    "verb": "pour",
                    "from": src,
                    "to": "waste",
                    "step_index": m.get("step_index"),
                }
            if m.get("verb") in ALLOWED:
                new.append(m)
        st["micro_ops"] = new


def _flatten_micro_plan_from_steps(doc: Dict[str, Any]) -> None:
    mp: List[Dict[str, Any]] = []
    for i, st in enumerate(doc.get("steps", []) or [], start=1):
        for m in st.get("micro_ops") or []:
            if m.get("verb") in ALLOWED:
                mm = dict(m)
                mm["step_index"] = i
                mp.append(mm)
    # de-dup consecutive identical ops
    out: List[Dict[str, Any]] = []
    for m in mp:
        if not out or out[-1] != m:
            out.append(m)
    doc["micro_plan"] = out


def polish_robot_doc(
    doc: Dict[str, Any], config: Dict[str, Any] | None = None
) -> Dict[str, Any]:
    cfg = {**DEFAULT_CONFIG, **(config or {})}
    # 0) mild cleanup
    _clean_reagent_names(doc)
    # 1) materials → vessels mapping
    _map_materials_to_vessels(doc, cfg)
    # 2) heat placement
    _normalize_heating_placement(doc, cfg)
    # 3) centrifuge canonicalization (+ optional wash cycles)
    _canonicalize_centrifuge(doc, cfg)
    # 4) drying coherence
    _enforce_drying(doc, cfg)
    # 5) base verbs only for steps
    _enforce_base_verbs(doc)
    # 6) registry cover
    _ensure_registry(doc, cfg)
    # 7) final flatten
    _flatten_micro_plan_from_steps(doc)
    return doc


# ---- CLI ----
if __name__ == "__main__":
    import json
    import pathlib
    import sys

    if len(sys.argv) < 2:
        print("Usage: python post_polish.py input.json [output.json]")
        sys.exit(2)
    inp = pathlib.Path(sys.argv[1])
    out = (
        pathlib.Path(sys.argv[2])
        if len(sys.argv) > 2
        else inp.with_suffix(".polished.json")
    )
    with inp.open("r", encoding="utf-8") as f:
        d = json.load(f)
    d = polish_robot_doc(d, config=None)
    with out.open("w", encoding="utf-8") as f:
        json.dump(d, f, indent=2)
    print("Wrote", out)
