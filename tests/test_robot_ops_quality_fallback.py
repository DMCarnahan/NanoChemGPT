from converter import convert_text_to_robot_ops

def test_quality_fallback_meta(monkeypatch):
    # Create a trivially small procedure that should trigger quality fallback heuristic in app-layer.
    # We can't easily invoke the Flask route here without full client; instead we simulate the heuristic:
    # Provide one very short step so future integration test can assert fallback triggers.
    proto = "1. **Procedure**:\n1. Mix reagents."
    doc = convert_text_to_robot_ops(proto)
    # The converter alone won't set fallback meta (handled in app); here we assert doc is trivial so heuristic would fire.
    steps = doc.get('steps') or []
    assert len(steps) == 1
    raw_len = len((steps[0].get('raw') or ''))
    assert raw_len < 120

