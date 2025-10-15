# Developer Notes: `converter.py`

## Overview
`converter.py` is the core module for converting free-text nanochemistry synthesis procedures into structured robot operations. It powers both micro_plan and minimal plan generation, supporting protocol normalization, device mapping, and postprocessing.

---

## Key Concepts

### 1. Micro Plan
- **Purpose:** Fine-grained extraction of all possible steps, including ambiguous or overlapping actions.
- **Logic:** Uses regex and NER to extract actions, amounts, devices, and conditions from text.
- **Output:** List of step dicts with raw and normalized fields.

### 2. Minimal Plan
- **Purpose:** Collapses redundant or consecutive steps into a minimal, executable protocol for robots.
- **Logic:**
  - Tracks device state (oven, hotplate, autotitrator, etc.)
  - Collapses temperature set operations globally (not just consecutively)
  - Deduplicates solution and solvent additions
  - Converts add_solvent and set_oven_temperature to robot-compatible operations
- **Output:** Ordered list of robot operations (pick_up, pour, set, heat, mix, wait, etc.)

### 3. Temperature Collapse
- **Old Logic:** Only collapsed consecutive set operations
- **New Logic:** Tracks all set operations globally using a `seen_sets` dictionary, ensuring only the last set for each device is kept
- **Benefit:** Prevents redundant temperature changes and ensures correct device state

### 4. Coverage Strategy
- **Scope:** Coverage is focused on `converter.py` for maintainability and speed
- **Rationale:** Other modules (harvester, retriever, etc.) are large and slow to test; converter is the critical path for protocol correctness
- **Test Types:**
  - Unit tests for regex extraction, device mapping, and postprocessing
  - Integration tests for full protocol conversion scenarios

### 5. Architectural Decisions
- **Regex-first extraction:** Fast and robust for domain-specific language
- **NER integration:** Custom spaCy model for material/entity extraction
- **Postprocessing:** Converts ambiguous or domain-specific steps to standardized robot actions
- **Error handling:** Graceful fallbacks for missing models or configs
- **Configurable via env and YAML:** Device registry, model paths, and index directories

---

## References
- See `tests/test_converter_full_scenario.py` for end-to-end protocol conversion tests
- See `pytest.ini` for coverage configuration
- See `.github/copilot-instructions.md` for architecture overview

---

_Last updated: 2025-10-15_
