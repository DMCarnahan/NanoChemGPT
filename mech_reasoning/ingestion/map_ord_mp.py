from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import time
from typing import Any, Dict, List

import requests

try:
    from ingestion.ingest_mechanisms import ingest
except Exception as e:
    raise SystemExit("ERROR: Could not import ingestion.ingest_mechanisms.") from e


# -------------------------- Helpers --------------------------


def _now_iso() -> str:
    return dt.datetime.utcnow().isoformat() + "Z"


def _coerce_num(x: Any) -> float | None:
    try:
        if x is None:
            return None
        return float(x)
    except Exception:
        return None


# -------------------------- Open Reaction Database (ORD) --------------------------

DEFAULT_ORD_URL = os.getenv(
    "ORD_API_URL", "https://api.open-reaction-database.org/records"
)


def fetch_ord(limit: int = 100, offset: int = 0) -> List[Dict[str, Any]]:
    """Fetch a page of ORD records (JSON)."""
    params = {"limit": max(1, min(limit, 1000)), "offset": max(0, offset)}
    r = requests.get(DEFAULT_ORD_URL, params=params, timeout=60)
    r.raise_for_status()
    data = r.json()
    for key in ("results", "records", "data", "items"):
        if isinstance(data, dict) and key in data and isinstance(data[key], list):
            return data[key]
    if isinstance(data, list):
        return data
    return []


def map_ord_record(rec: Dict[str, Any]) -> Dict[str, Any] | None:
    """Map one ORD record -> mechanistic entry dict."""
    system = None
    products = rec.get("products") or rec.get("outcomes") or []
    if isinstance(products, list) and products:
        p = products[0]
        system = p.get("smiles") or p.get("name") or p.get("product_name")
    system = (
        system or rec.get("title") or rec.get("reaction_name") or "Unknown reaction"
    )

    rxn_type = rec.get("reaction_type") or rec.get("type") or "solution-phase reaction"

    doi = ""
    prov = rec.get("provenance") or rec.get("provenances") or {}
    if isinstance(prov, dict):
        doi = prov.get("doi") or prov.get("DOI") or ""
    elif isinstance(prov, list) and prov:
        d0 = prov[0]
        if isinstance(d0, dict):
            doi = d0.get("doi") or d0.get("DOI") or ""

    params = []
    cond = rec.get("conditions") or rec.get("reaction_conditions") or {}

    def add_param(
        name: str,
        units: str,
        value: Any,
        role: str,
        target="yield",
        direction="unknown",
        rationale="",
    ):
        if value is None:
            return
        params.append(
            {
                "name": name,
                "units": units,
                "role": role,
                "effects": [
                    {
                        "target": target,
                        "direction": direction,
                        "mechanistic_rationale": rationale
                        or f"{name} can influence {target}",
                    }
                ],
            }
        )

    # temperature
    temp = None
    if isinstance(cond, dict):
        tblob = cond.get("temperature") or cond.get("temp") or {}
        if isinstance(tblob, dict):
            temp = _coerce_num(tblob.get("value") or tblob.get("degrees_celsius"))
    add_param("temperature", "°C", temp, "reaction condition", target="yield")

    # time
    t_min = None
    tblob = cond.get("time") if isinstance(cond, dict) else None
    if isinstance(tblob, dict):
        seconds = _coerce_num(tblob.get("seconds") or tblob.get("value"))
        if seconds is not None:
            t_min = seconds / 60.0
    add_param("time", "min", t_min, "reaction condition", target="yield")

    # solvent
    solvent = None
    if isinstance(cond, dict):
        s = cond.get("solvent") or cond.get("solvents")
        if isinstance(s, list) and s:
            solvent = s[0].get("name") if isinstance(s[0], dict) else str(s[0])
        elif isinstance(s, dict):
            solvent = s.get("name")
    if solvent:
        add_param(
            "solvent",
            "",
            solvent,
            "medium",
            target="yield",
            rationale="Solvent can alter activity/selectivity",
        )

    catalyst = rec.get("catalyst") or (
        cond.get("catalyst") if isinstance(cond, dict) else None
    )
    if catalyst:
        add_param(
            "catalyst",
            "",
            catalyst,
            "catalyst system",
            target="yield",
            rationale="Catalyst changes rate/selectivity",
        )

    observed_outcomes = []
    outcome = rec.get("outcome") or rec.get("outcomes") or {}
    yld = None
    if isinstance(outcome, dict):
        yld = _coerce_num(outcome.get("yield_percent") or outcome.get("yield"))
    elif isinstance(outcome, list) and outcome:
        o0 = outcome[0]
        if isinstance(o0, dict):
            yld = _coerce_num(o0.get("yield_percent") or o0.get("yield"))
    if yld is not None:
        observed_outcomes.append(
            {"metric": "yield", "value": yld, "units": "%", "notes": ""}
        )

    return {
        "system": system,
        "synthesis_method": rxn_type,
        "mechanisms": [{"name": rxn_type, "confidence": 0.5}],
        "parameters": params,
        "observed_outcomes": observed_outcomes,
        "most_influential_parameter": {},
        "evidence": (
            [
                {
                    "source_type": "paper",
                    "citation": doi or "",
                    "url": (f"https://doi.org/{doi}" if doi else ""),
                    "quote": "",
                }
            ]
            if doi
            else []
        ),
        "created_at": _now_iso(),
        "updated_at": _now_iso(),
    }


def harvest_ord(
    total_limit: int = 200, page_size: int = 100, sleep_s: float = 0.5
) -> List[Dict[str, Any]]:
    entries: List[Dict[str, Any]] = []
    fetched = 0
    while fetched < total_limit:
        batch = fetch_ord(limit=min(page_size, total_limit - fetched), offset=fetched)
        if not batch:
            break
        for rec in batch:
            try:
                e = map_ord_record(rec)
                if e:
                    entries.append(e)
            except Exception:
                continue
        fetched += len(batch)
        if len(batch) < page_size:
            break
        time.sleep(sleep_s)
    return entries


# -------------------------- Materials Project (MP) --------------------------

DEFAULT_MP_URL = os.getenv(
    "MP_API_URL", "https://api.materialsproject.org/materials/summary"
)
MP_API_KEY = os.getenv("MP_API_KEY", "")


def fetch_mp_by_formulas(
    formulas: List[str], per_formula_limit: int = 100
) -> List[Dict[str, Any]]:
    if not MP_API_KEY:
        raise RuntimeError("MP_API_KEY is not set. Export MP_API_KEY first.")
    headers = {"X-API-KEY": MP_API_KEY}
    out: List[Dict[str, Any]] = []
    for formula in formulas:
        params = {"formula": formula, "chunk_size": min(500, per_formula_limit)}
        r = requests.get(DEFAULT_MP_URL, headers=headers, params=params, timeout=60)
        r.raise_for_status()
        data = r.json()
        recs = data.get("data") or data.get("results") or []
        out.extend(recs[:per_formula_limit])
    return out


def map_mp_record(rec: Dict[str, Any]) -> Dict[str, Any] | None:
    system = (
        rec.get("formula_pretty")
        or rec.get("full_formula")
        or rec.get("composition")
        or "Unknown material"
    )
    dois = []
    bibs = rec.get("references") or rec.get("bibtex") or []
    if isinstance(bibs, list):
        for b in bibs:
            if isinstance(b, dict) and b.get("doi"):
                dois.append(b["doi"])
    elif isinstance(bibs, dict):
        if bibs.get("doi"):
            dois.append(bibs["doi"])

    parameters = [
        {
            "name": "annealing temperature",
            "units": "°C",
            "role": "sets phase formation thermodynamics",
            "effects": [
                {
                    "target": "phase_purity",
                    "direction": "increase",
                    "mechanistic_rationale": "Higher temperature improves diffusion and phase ordering in solids",
                }
            ],
        },
        {
            "name": "annealing time",
            "units": "h",
            "role": "controls grain growth",
            "effects": [
                {
                    "target": "grain_size",
                    "direction": "increase",
                    "mechanistic_rationale": "Longer anneals allow grains to grow",
                }
            ],
        },
    ]

    return {
        "system": system,
        "phase": (rec.get("symmetry") or {}).get("symbol", ""),
        "synthesis_method": "solid-state (inferred)",
        "route_details": "Derived from Materials Project summary",
        "mechanisms": [{"name": "thermodynamic control", "confidence": 0.6}],
        "parameters": parameters,
        "observed_outcomes": [
            {
                "metric": "formation_energy_per_atom",
                "value": rec.get("formation_energy_per_atom", ""),
                "units": "eV/atom",
            },
            {"metric": "band_gap", "value": (rec.get("band_gap") or ""), "units": "eV"},
        ],
        "most_influential_parameter": {},
        "evidence": (
            [
                {
                    "source_type": "paper",
                    "citation": dois[0],
                    "url": f"https://doi.org/{dois[0]}",
                    "quote": "",
                }
            ]
            if dois
            else []
        ),
        "created_at": _now_iso(),
        "updated_at": _now_iso(),
    }


# -------------------------- Main CLI --------------------------


def main():
    ap = argparse.ArgumentParser(description="Ingest ORD and MP into mechanistic_kb")
    ap.add_argument("--ord", action="store_true", help="Harvest Open Reaction Database")
    ap.add_argument(
        "--mp", action="store_true", help="Harvest Materials Project summaries"
    )
    ap.add_argument("--ord-limit", type=int, default=200)
    ap.add_argument("--ord-page", type=int, default=100)
    ap.add_argument("--mp-formulas", type=str, default="")
    ap.add_argument("--mp-limit", type=int, default=100)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    all_entries: List[Dict[str, Any]] = []

    if args.ord:
        ord_entries = harvest_ord(total_limit=args.ord_limit, page_size=args.ord_page)
        print(f"[ORD] Mapped entries: {len(ord_entries)}")
        all_entries.extend(ord_entries)

    if args.mp:
        if not args.mp_formulas.strip():
            raise SystemExit("ERROR: --mp requires --mp-formulas 'A,B'")
        formulas = [s.strip() for s in args.mp_formulas.split(",") if s.strip()]
        mp_raw = fetch_mp_by_formulas(formulas, per_formula_limit=args.mp_limit)
        mp_entries = [e for e in (map_mp_record(r) for r in mp_raw) if e]
        print(f"[MP] Mapped entries: {len(mp_entries)}")
        all_entries.extend(mp_entries)

    if not all_entries:
        print("No entries to ingest.")
        return

    if args.dry_run:
        print(f"[DRY-RUN] Would ingest {len(all_entries)} entries")
        print(json.dumps(all_entries[0], indent=2))
        return

    ids = ingest(all_entries)
    print(f"[INGESTED] {len(ids)} entries into mechanistic_kb")


if __name__ == "__main__":
    main()
