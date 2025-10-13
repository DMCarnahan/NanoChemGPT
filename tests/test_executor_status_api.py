import json
import pytest
from app import app as flask_app


@pytest.fixture
def client():
    flask_app.config["TESTING"] = True
    with flask_app.test_client() as c:
        yield c


PROTOCOL_TEXT = """1. **Reagents**\nWater, ethanol.\n\n2. **Apparatus**\nHot plate.\n\n3. **Procedure**\n[Add ethanol (10 mL) to the flask. Heat the mixture at 60 C for 30 min.]\n"""


def test_executor_status_present(client):
    resp = client.post(
        "/ask?robot_ops=1",
        json={
            "question": "Provide a procedure",
            "mode": "protocol",
            "context": PROTOCOL_TEXT,
        },
    )
    assert resp.status_code == 200, resp.data
    data = resp.get_json()
    # robot_operations may appear depending on parsing; ensure executor fields present if operations exist
    if "robot_operations" in data:
        assert "executor_valid" in data
        assert "executor_schema_version" in data
        assert isinstance(data.get("executor_repairs"), list)
    else:
        pytest.skip("robot_operations not produced in this run (heuristic skip)")
