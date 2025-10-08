import os
import json
from converter import convert_text_to_robot_ops

SAMPLE = """1. **Procedure**:\n1. Dissolve 1.0 g NaCl in 10 mL water.\n2. Add 5 mL ethanol to the solution.\n3. Heat the mixture to 60 C for 30 minutes.\n4. Dry the product in an oven at 80 C for 2 h."""

def _get():
    return convert_text_to_robot_ops(SAMPLE)

def test_min_plan_primitives_only():
    doc = _get()
    assert 'micro_plan_min' in doc, 'minimal plan missing'
    for a in doc['micro_plan_min']:
        assert a['verb'] in {'pick_up','place','pour','set'}, f"Unexpected verb {a['verb']}"


def test_min_plan_delays_present():
    doc = _get()
    # Expect at least one delay because of heat (30 min) and dry (120 min)
    delays = doc.get('timing_delays')
    assert delays and any(d.get('minutes') == 30 for d in delays), '30 min delay missing'


def test_min_plan_device_mapping():
    os.environ['MIN_PLAN_MAP_GENERIC'] = '1'
    try:
        doc = _get()
        # At least one set action should map HP1->hotplate
        sets = [a for a in doc['micro_plan_min'] if a['verb']=='set']
        assert any(a.get('device')=='hotplate' for a in sets), 'Device mapping did not apply'
    finally:
        os.environ.pop('MIN_PLAN_MAP_GENERIC', None)
