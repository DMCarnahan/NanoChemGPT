import json
import re

from app import app as flask_app


def test_robot_ops_flag_roundtrip(monkeypatch):
    client = flask_app.test_client()

    # Mock OpenAI client behavior by monkeypatching the chat completion call
    class FakeChoice:
        def __init__(self, content):
            self.message = type('m', (), {'content': content})

    class FakeResp:
        def __init__(self, content):
            self.choices = [FakeChoice(content)]

    class FakeChat:
        def completions(self, *a, **k):
            raise NotImplementedError()
        def create(self, *a, **k):
            # Return minimal protocol block matching expected scaffold
            protocol = (
                "## Synthesis Protocol:\n"
                "1. **Hardware & Glassware**:\n[]\n"
                "2. **Materials**:\n[]\n"
                "3. **Procedure**\n[Add 5 mg cobalt nitrate to Beaker. Heat to 80 °C for 30 min.]\n\n"
                "```reason\nRationale here.\n```\n"
            )
            return FakeResp(protocol)

    class FakeClient:
        def __init__(self):
            self.chat = type('chat', (), {'completions': type('c', (), {'create': FakeChat().create})})()

    # Inject fake client
    import app as app_module
    monkeypatch.setattr(app_module, 'client', FakeClient())

    resp = client.post('/ask?robot_ops=1', json={'question': 'Give a cobalt nanowire synthesis protocol'})
    assert resp.status_code == 200
    data = resp.get_json()
    assert data.get('ok') is True
    assert 'robot_operations' in data, f"robot_operations missing; keys={list(data.keys())}"
    assert 'robot_rules' in data, "robot_rules alias missing"
    ops = data['robot_operations']
    assert isinstance(ops, dict), 'robot_operations should be dict returned by convert_text_to_robot_ops'
    # Expect at least one inferred record (depending on parser heuristics). Allow empty but present.
    assert 'records' in ops or len(ops.keys()) > 0
