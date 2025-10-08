import os
from converter import convert_text_to_robot_ops

SAMPLE = """1. **Procedure**:\n1. Dissolve 0.5 g FeSO4·7H2O in 25 mL deionized water.\n2. Add 5 mL ethanol to the solution.\n3. Transfer the mixture to a clean beaker.\n4. Heat the mixture to 60 C for 30 minutes.\n5. Dry the product in an oven at 80 C for 2 h."""

def _convert():
    return convert_text_to_robot_ops(SAMPLE)


def test_min_plan_has_pour_for_additions_and_transfer():
    doc = _convert()
    assert 'micro_plan_min' in doc, 'missing minimal plan'
    verbs = [a['verb'] for a in doc['micro_plan_min']]
    assert 'pour' in verbs, 'expected a pour action when transfer/add ops present'


def test_no_zero_minute_delays():
    doc = _convert()
    for d in doc.get('timing_delays', []):
        assert d.get('minutes') != 0, 'zero-minute delay should be suppressed'


def test_step_indices_present():
    doc = _convert()
    for a in doc['micro_plan_min']:
        assert a.get('step_index') is not None, 'step_index should be backfilled'


def test_oven_set_present_in_min_plan():
    doc = _convert()
    # If drying step exists, ensure a set temperature for oven is present
    if any('dry' in (s.get('raw','').lower()) for s in doc.get('steps', [])):
        assert any(a.get('device') in {'OV1','oven'} and a.get('param')=='temperature_C' for a in doc['micro_plan_min']), 'oven temperature set missing in minimal plan'


def test_reagent_name_not_truncated():
    doc = _convert()
    # FeSO4·7H2O line should retain solute name in reagents or structured
    dissolve_step = next((s for s in doc['steps'] if s.get('action')=='dissolve'), None)
    assert dissolve_step, 'dissolve step missing'
    assert any('FeSO4' in (r.get('name') if isinstance(r, dict) else r) for r in dissolve_step.get('reagents_structured', [])), 'FeSO4·7H2O name truncated'
