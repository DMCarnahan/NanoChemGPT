from converter import convert_text_to_robot_ops


def test_oven_pickup_before_place():
    text = "Dry the sample in an oven at 50 C for 1 hour."
    doc = convert_text_to_robot_ops(text)
    mp = doc.get("micro_plan", [])
    places = [i for i,a in enumerate(mp) if a.get('verb')=='place' and a.get('to') in {'OV1','oven'}]
    assert places, "No oven placement found"
    oven_idx = places[0]
    assert oven_idx>0 and mp[oven_idx-1].get('verb')=='pick_up', "Oven place not preceded by pick_up"


def test_naming_unification():
    text = "Add reagent A to vessel V3 while stirring. Then heat vessel V3."
    doc = convert_text_to_robot_ops(text)
    # ensure no v3_bottle lingering if V3 recognized
    assert not any('v3_bottle' == a.get('object') for a in doc.get('micro_plan', []))


def test_autotitrator_rate_set():
    text = "Using an autotitrator, add reagent B to vessel V3 at a controlled rate of 5 mL/min while stirring for 10 minutes."
    doc = convert_text_to_robot_ops(text)
    # look for a set op with rate param (canonical or legacy)
    has_rate = (
        any(a.get('verb')=='set' and a.get('param') in {'rate_mL_per_min','rate_ml_per_min'} for a in doc.get('micro_plan', []))
        or any(op.get('op')=='set' and op.get('param') in {'rate_mL_per_min','rate_ml_per_min'} for st in doc.get('steps', []) for op in (st.get('ops') or []))
    )
    assert has_rate


def test_cross_step_idempotent_set_collapse():
    text = (
        "Heat the mixture to 45 C for 5 minutes. "
        "Continue heating at 45 C for 3 minutes. "
        "Maintain temperature at 45 C while adding reagent C."
    )
    doc = convert_text_to_robot_ops(text)
    min_plan = doc.get('micro_plan_min', [])
    # only one set temperature_C=45 expected
    sets = [a for a in min_plan if a.get('verb')=='set' and a.get('param')=='temperature_C' and a.get('value')==45]
    assert len(sets)==1
    # provenance should include multiple step indices
    if 'collapsed_from_steps' in sets[0]:
        assert len(sets[0]['collapsed_from_steps'])>=2


def test_no_minutes_zero_key():
    text = "Add reagent D to vessel V3."
    doc = convert_text_to_robot_ops(text)
    for st in doc.get('steps', []):
        assert st.get('minutes') != 0, st
