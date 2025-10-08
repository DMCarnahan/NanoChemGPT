import json
from converter import convert_text_to_robot_ops


def test_freeform_drying_pickup_place():
    doc = convert_text_to_robot_ops("Dry sample in oven at 60 C for 30 minutes.")
    mp = doc.get("micro_plan", [])
    places = [i for i,a in enumerate(mp) if a.get('verb')=='place' and a.get('to') in {'OV1','oven'}]
    assert places, 'No oven placement found'
    idx = places[0]
    assert idx>0 and mp[idx-1].get('verb')=='pick_up'
    assert any(a.get('verb')=='set' and a.get('device') in {'OV1','oven'} for a in mp)


def test_structured_drying_numbered():
    text = """Procedure:\n1. Heat the mixture to 50 C.\n2. Dry the sample in an oven at 50 C for 1 hour.\n"""
    doc = convert_text_to_robot_ops(text)
    mp = doc.get('micro_plan', [])
    oven_sets = [a for a in mp if a.get('verb')=='set' and a.get('device') in {'OV1','oven'} and a.get('param')=='temperature_C']
    assert oven_sets, 'Expected oven temperature set'
    places = [i for i,a in enumerate(mp) if a.get('verb')=='place' and a.get('to') in {'OV1','oven'}]
    assert places and places[0]>0 and mp[places[0]-1].get('verb')=='pick_up'


def test_autotitrator_variant_phrase():
    doc = convert_text_to_robot_ops('Autotitrator-assisted addition: add solution A to vessel V2 at 2.5 mL/min while stirring for 5 minutes.')
    assert any(a.get('verb')=='set' and a.get('param')=='rate_ml_per_min' and a.get('value')==2.5 for a in doc.get('micro_plan', []))


def test_temperature_set_provenance_collapse():
    text = 'Heat to 45 C. Continue heating at 45 C. Maintain temperature at 45 C.'
    doc = convert_text_to_robot_ops(text)
    min_plan = doc.get('micro_plan_min', [])
    sets = [a for a in min_plan if a.get('verb')=='set' and a.get('param')=='temperature_C' and a.get('value')==45]
    assert len(sets)==1, 'Expected collapsed single temperature set'
    # If provenance recorded, must show multiple steps
    prov = sets[0].get('collapsed_from_steps')
    if prov:
        assert len(prov)>=2
