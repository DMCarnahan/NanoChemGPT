from pathlib import Path

from harvester import oa_resolver


class _Response:
    def raise_for_status(self):
        return None

    def json(self):
        return {}


class _Session:
    def __init__(self):
        self.calls = []

    def get(self, url, **kwargs):
        self.calls.append((url, kwargs))
        return _Response()


def test_openalex_credentials_are_optional_and_blank_values_are_omitted(monkeypatch):
    session = _Session()
    monkeypatch.setenv("OPENALEX_API_KEY", "   ")
    monkeypatch.setenv("OPENALEX_MAILTO", "")

    assert oa_resolver.openalex_get_by_doi("10.1000/example", session=session) == {}
    assert session.calls[0][1]["params"] == {}


def test_openalex_credentials_are_forwarded_when_configured(monkeypatch):
    session = _Session()
    monkeypatch.setenv("OPENALEX_API_KEY", "openalex-test-key")
    monkeypatch.setenv("OPENALEX_MAILTO", "researcher@example.com")

    oa_resolver.openalex_get_by_doi("10.1000/example", session=session)

    assert session.calls[0][1]["params"] == {
        "api_key": "openalex-test-key",
        "mailto": "researcher@example.com",
    }


def test_unpaywall_blank_email_skips_the_request():
    session = _Session()

    assert oa_resolver.unpaywall_get("10.1000/example", "", session=session) == {}
    assert session.calls == []


def test_blank_or_invalid_oa_timeout_uses_default(monkeypatch):
    monkeypatch.setenv("OA_TIMEOUT", "")
    assert oa_resolver._env_float("OA_TIMEOUT", 12.0) == 12.0

    monkeypatch.setenv("OA_TIMEOUT", "not-a-number")
    assert oa_resolver._env_float("OA_TIMEOUT", 12.0) == 12.0


def test_optional_literature_variables_are_documented():
    env_example = (Path(__file__).resolve().parents[1] / "env.example").read_text(
        encoding="utf-8"
    )

    for name in (
        "OPENALEX_API_KEY=",
        "OPENALEX_MAILTO=",
        "UNPAYWALL_EMAIL=",
        "EPMC_BASE=",
    ):
        assert name in env_example
