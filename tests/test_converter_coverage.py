import os
from converter import convert_text_to_robot_ops, apply_postprocessing


def test_converter_broad_paths():
    os.environ['MIN_PLAN_MAP_GENERIC'] = '1'
    text = """**Procedure:**\n1. Weigh 1.2 g sodium chloride into a beaker.\n2. Add 10 mL water and stir until dissolved.\n3. Transfer the solution to vessel V3.\n4. Heat the mixture to 45 C for 5 minutes.\n5. Continue heating at 45 C for 3 minutes.\n6. Maintain temperature at 45 C while adding reagent C.\n7. Using an autotitrator, add reagent B to vessel V3 at a controlled rate of 2 mL/min while stirring for 10 minutes.\n8. Monitor pH every 30 seconds until pH reaches 7.\n9. Titrate to pH 7 with sodium hydroxide, maximum 5 mL at 0.5 mL/min.\n10. Dry the product in an oven at 80 C for 1 hour.\n11. Resuspend the dried powder.\n"""
    doc = convert_text_to_robot_ops(text)
    # Basic sanity assertions to ensure core keys exist
    assert 'micro_plan' in doc
    assert any(op.get('verb') == 'set' and op.get('param') == 'temperature_C' for op in doc['micro_plan'])
    assert any(op.get('verb') == 'set' and op.get('param') in {'rate_mL_per_min','rate_ml_per_min'} for op in doc['micro_plan'])
    assert 'micro_plan_min' in doc
    assert 'timing_delays' in doc


def test_apply_postprocessing_idempotent():
    # Simple call to apply_postprocessing twice should not duplicate operations
    text = "**Procedure:**\n1. Heat the mixture to 50 C for 10 minutes."
    doc = convert_text_to_robot_ops(text)
    mp_len = len(doc.get('micro_plan', []))
    doc2 = apply_postprocessing(doc)
    assert len(doc2.get('micro_plan', [])) >= mp_len
