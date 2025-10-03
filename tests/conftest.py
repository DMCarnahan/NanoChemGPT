import pytest


class DummyChoice:
    def __init__(self, content):
        self.choices = [
            type("C", (), {"message": type("M", (), {"content": content})})()
        ]


class DummyClient:
    def __init__(self, reply_text):
        self._reply = reply_text

    class chat:
        @staticmethod
        def completions_create_stub(*args, **kwargs):
            # this will be replaced by instance method in fixture
            return None

    def chat_completions_create(self, *args, **kwargs):
        return type(
            "R",
            (),
            {
                "choices": [
                    type(
                        "C", (), {"message": type("M", (), {"content": self._reply})}
                    )()
                ]
            },
        )()


@pytest.fixture(autouse=True)
def fake_openai_client(monkeypatch, request):
    """Replace `client` in app module with a dummy client for tests so `/ask` returns a deterministic reply."""
    # If this is an integration test (marked with @pytest.mark.integration),
    # skip importing the whole `app` module to avoid heavy imports like faiss.
    if request.node.get_closest_marker("integration"):
        # do nothing for integration tests; they exercise miner directly.
        yield
        return
    try:
        import app
    except Exception:
        # If the full `app` module (and its heavy deps) are not available in
        # the test environment, create a minimal dummy module that provides
        # the `client` attribute used by tests. This avoids needing Flask,
        # httpx, etc. for pure unit tests.
        import types

        app = types.SimpleNamespace()
        app.client = None

    class FakeResp:
        def __init__(self, content):
            class Msg:
                def __init__(self, c):
                    self.content = c

            self.choices = [type("C", (), {"message": Msg(content)})()]

    class FakeClient:
        def __init__(self):
            pass

        class chat:
            @staticmethod
            def completions_create(**kwargs):
                # Return a simple answer that doesn't require context
                return FakeResp("This is a fake answer.\n\n## References\n")

    monkeypatch.setattr(app, "client", FakeClient())
    yield
    # teardown handled by monkeypatch
