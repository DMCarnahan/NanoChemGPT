from converter import convert_text_to_robot_ops

FULL_TEXT = """**Procedure:**
1. Weigh 1.0 g precursor A into a beaker.
2. Prepare a 0.5 M solution by dissolving 0.5 g salt B in 5 mL water.
3. Add 3 mL ethanol then add 2 mL water to the beaker while stirring for 1 minutes.
4. Sonicate the suspension for 2 minutes.
5. Heat the mixture to 50 C for 4 minutes.
6. Continue heating at 50 C for 2 minutes.
7. Maintain temperature at 50 C while stirring vigorously.
8. Cool the solution to 20 C.
9. Dissolve any remaining solid completely.
10. Filter the mixture and wash with water then dry.
11. Collect the solid and discard the filtrate.
12. Transfer the solution to vessel V2.
13. Dry the product in an oven at 90 C for 30 minutes.
14. Resuspend the dried powder.
15. Monitor pH every 45 seconds until pH reaches 7.
16. Titrate to pH 7 with sodium hydroxide, maximum 4 mL at 0.4 mL/min.
"""

SINGLE_LINE_FALLBACK = "Using an autotitrator, add reagent C to vessel V3 at a controlled rate of 1 mL/min while stirring for 5 minutes."

def test_full_scenario_and_fallback():
    doc1 = convert_text_to_robot_ops(FULL_TEXT)
    doc2 = convert_text_to_robot_ops(SINGLE_LINE_FALLBACK)
    # Basic assertions: micro_plan present and has key verbs
    verbs1 = {op.get('verb') for op in doc1.get('micro_plan', [])}
    assert {'set','pick_up','place','wait'} & verbs1
    # Temperature collapse: only one temperature set in minimal plan for 50 C
    min_sets = [a for a in doc1.get('micro_plan_min', []) if a.get('verb')=='set' and a.get('param')=='temperature_C' and a.get('value')==50]
    assert len(min_sets) == 1
    # Autotitrator rate in second doc
    assert any(op.get('verb')=='set' and op.get('param') in {'rate_mL_per_min','rate_ml_per_min'} for op in doc2.get('micro_plan', []))
    # Timing delays exist
    assert doc1.get('timing_delays') is not None
