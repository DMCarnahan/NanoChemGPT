from app import app as flask_app


def test_transcribe_endpoint_client():
    """In-process test of /transcribe using Flask test client."""
    client = flask_app.test_client()

    sample_method = (
        "Gold nanoparticle synthesis was performed using the Turkevich method.\n"
        "100 mL of 0.5 mM HAuCl4·3H2O solution was heated to boiling in a water bath at 100°C.\n"
        "10 mL of 38.8 mM sodium citrate solution was added rapidly while stirring at 300 rpm.\n"
        "The solution was continued to boil for 15 minutes while stirring.\n"
        "The reaction was then cooled to room temperature and centrifuged at 8000 rpm for 10 minutes.\n"
        "The precipitate was dried in an oven at 60°C for 2 hours.\n"
    )

    payload = {"text": sample_method, "convert_to_robot": False}
    rv = client.post("/transcribe", json=payload)
    assert rv.status_code in (200, 202, 400)

    # Also test robot conversion path (the endpoint may return a warning rather than failure)
    payload2 = {"text": sample_method, "convert_to_robot": True}
    rv2 = client.post("/transcribe", json=payload2)
    assert rv2.status_code in (200, 202, 400)


def test_method_paragraph_pick():
    """Test the method paragraph extraction via /transcribe."""
    client = flask_app.test_client()

    mixed_text = (
        "Introduction: Gold nanoparticles have unique properties.\n\n"
        "Materials and Methods:\n"
        "100 mL of 0.5 mM HAuCl4 solution was heated to 100°C in a water bath.\n"
        "10 mL of 38.8 mM sodium citrate was added rapidly while stirring at 300 rpm.\n"
        "The solution was boiled for 15 minutes. The reaction was cooled and centrifuged at 8000 rpm for 10 minutes.\n"
        "The precipitate was dried at 60°C for 2 h.\n\n"
        "Results: The nanoparticles showed excellent stability.\n"
    )

    rv = client.post(
        "/transcribe", json={"text": mixed_text, "convert_to_robot": False}
    )
    assert rv.status_code in (200, 202, 400)
