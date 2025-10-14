from converter import convert_text_to_robot_ops, apply_postprocessing

# Exercise weigh + dissolve + heat + cool + filter + wash/dry + oven + resuspend + transfer explicit

def test_converter_branch_mix():
    text = """**Procedure:**\n1. Weigh 0.5 g compound X into a beaker.\n2. Add 5 mL ethanol and 10 mL water and stir for 2 minutes.\n3. Dissolve the solid completely.\n4. Heat the mixture to 60 C for 5 minutes.\n5. Cool the solution to 20 C.\n6. Filter the mixture and wash with water then dry.\n7. Dry the product in an oven at 80 C for 30 minutes.\n8. Resuspend the dried powder.\n9. Transfer the solution to vessel V2.\n10. Continue heating at 60 C for 3 minutes.\n"""
    doc = convert_text_to_robot_ops(text)
    mp = doc.get('micro_plan', [])
    # Assertions to ensure representative verbs appear
    verbs = {op.get('verb') for op in mp}
    expected = {'pick_up','place','wait','set','pour'}
    assert expected & verbs, f"Missing expected verbs: {expected - verbs}"
    # Ensure minimal plan present
    assert 'micro_plan_min' in doc


def test_apply_postprocessing_rate_and_temp_collapse():
    text = ("Heat the mixture to 45 C for 1 minutes. Continue heating at 45 C for 2 minutes. "
            "Maintain temperature at 45 C while stirring.")
    doc = convert_text_to_robot_ops(text)
    # Collapsed minimal plan should have only one temperature set with provenance
    mp_min = doc.get('micro_plan_min', [])
    sets = [a for a in mp_min if a.get('verb')=='set' and a.get('param')=='temperature_C' and a.get('value')==45]
    assert len(sets) == 1
    # micro_plan should retain a set
    assert any(op.get('verb')=='set' and op.get('param')=='temperature_C' for op in doc.get('micro_plan', []))
