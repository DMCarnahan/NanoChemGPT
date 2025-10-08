import json
from converter import convert_text_to_robot_ops


def test_no_zero_minute_waits():
    text = """
    Heat the solution to 50 C for 30 minutes.
    Add reagent A with immediate mixing.
    Dry at 60 C for 1 hour.
    """
    doc = convert_text_to_robot_ops(text)
    # Ensure no wait with minutes == 0 anywhere
    for m in doc.get("micro_plan", []):
        assert not (m.get("verb") == "wait" and int(m.get("minutes") or 0) == 0), m
    for st in doc.get("steps", []):
        for m in st.get("micro_ops", []) or []:
            assert not (m.get("verb") == "wait" and int(m.get("minutes") or 0) == 0), (st.get("index"), m)
        for op in st.get("ops", []) or []:
            assert not (op.get("op") == "wait" and int(op.get("minutes") or 0) == 0), (st.get("index"), op)
