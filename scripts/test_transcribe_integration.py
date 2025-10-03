"""
Test script for the new /transcribe endpoint integration
"""

import requests


def test_transcribe_endpoint():
    """Test the new transcribe endpoint with sample method text."""

    # Sample method paragraph from a nanochemistry paper
    sample_method = """
    Gold nanoparticle synthesis was performed using the Turkevich method. 
    100 mL of 0.5 mM HAuCl₄·3H₂O solution was heated to boiling in a water bath at 100°C. 
    10 mL of 38.8 mM sodium citrate solution was added rapidly while stirring at 300 rpm. 
    The solution was continued to boil for 15 minutes while stirring. 
    The reaction was then cooled to room temperature and centrifuged at 8000 rpm for 10 minutes.
    The precipitate was dried in an oven at 60°C for 2 hours.
    """

    base_url = "http://localhost:5000"

    print("Testing /transcribe endpoint...")

    # Test 1: Direct text input without robot conversion
    payload1 = {"text": sample_method, "convert_to_robot": False}

    try:
        response1 = requests.post(f"{base_url}/transcribe", json=payload1, timeout=30)
        print(f"Test 1 - Status: {response1.status_code}")

        if response1.status_code == 200:
            result1 = response1.json()
            print("✅ Structured protocol generated successfully")
            print(
                "Extracted facts:",
                len(result1.get("extracted_facts", {}).get("materials", [])),
                "materials",
            )
            print(
                "Procedure steps:",
                len(result1.get("extracted_facts", {}).get("procedure", [])),
                "steps",
            )
        else:
            print(f"❌ Test 1 failed: {response1.text}")

    except Exception as e:
        print(f"❌ Test 1 error: {e}")

    # Test 2: Direct text input with robot conversion
    payload2 = {"text": sample_method, "convert_to_robot": True}

    try:
        response2 = requests.post(f"{base_url}/transcribe", json=payload2, timeout=30)
        print(f"\nTest 2 - Status: {response2.status_code}")

        if response2.status_code == 200:
            result2 = response2.json()
            print("✅ Robot operations generated successfully")
            if "robot_operations" in result2:
                print("Robot operations included in response")
            if "robot_conversion_error" in result2:
                print(
                    f"⚠️ Robot conversion warning: {result2['robot_conversion_error']}"
                )
        else:
            print(f"❌ Test 2 failed: {response2.text}")

    except Exception as e:
        print(f"❌ Test 2 error: {e}")

    # Test 3: Compare with old structured mode via /ask
    print("\nTesting integration with /ask endpoint...")

    # Create a test file
    import os
    import tempfile

    # Ensure we write the temp file with UTF-8 encoding on Windows
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".txt", delete=False, encoding="utf-8"
    ) as f:
        f.write(sample_method)
        temp_file_path = f.name

    try:
        with open(temp_file_path, "rb") as f:
            files = {"file": ("method.txt", f, "text/plain")}
            data = {"question": "transcribe the procedure verbatim"}

            response3 = requests.post(
                f"{base_url}/ask", data=data, files=files, timeout=30
            )
            print(f"Test 3 - /ask endpoint status: {response3.status_code}")

            if response3.status_code == 200:
                result3 = response3.json()
                if result3.get("ok"):
                    print("✅ Enhanced /ask endpoint working")
                    if "method_paragraph_used" in result3:
                        print(
                            f"Method paragraph extraction: {result3['method_paragraph_used']}"
                        )
                else:
                    print(f"❌ /ask failed: {result3.get('error', 'Unknown error')}")
            else:
                print(f"❌ Test 3 failed: {response3.text}")

    except Exception as e:
        print(f"❌ Test 3 error: {e}")
    finally:
        # Clean up temp file
        if os.path.exists(temp_file_path):
            os.unlink(temp_file_path)


def test_method_paragraph_extraction():
    """Test the _pick_method_paragraph function indirectly."""

    # Sample text with multiple paragraphs, only one being a method
    mixed_text = """
    Introduction: Gold nanoparticles have unique properties.
    
    Materials and Methods: 
    100 mL of 0.5 mM HAuCl₄ solution was heated to 100°C in a water bath.
    10 mL of 38.8 mM sodium citrate was added rapidly while stirring at 300 rpm.
    The solution was boiled for 15 minutes. The reaction was cooled and 
    centrifuged at 8000 rpm for 10 minutes. The precipitate was dried at 60°C for 2 h.
    
    Results: The nanoparticles showed excellent stability.
    """

    base_url = "http://localhost:5000"

    payload = {"text": mixed_text, "convert_to_robot": False}

    try:
        response = requests.post(f"{base_url}/transcribe", json=payload, timeout=30)

        if response.status_code == 200:
            result = response.json()
            original_length = len(mixed_text)
            processed_length = len(result.get("original_text", ""))

            print(f"\nMethod extraction test:")
            print(f"Original text: {original_length} chars")
            print(f"Processed text: {processed_length} chars")

            if processed_length < original_length:
                print("✅ Method paragraph extraction working - text was filtered")
            else:
                print(
                    "ℹ️ Full text used (method extraction may not have found a specific paragraph)"
                )

            # Check if structured protocol was generated
            if result.get("structured_protocol"):
                print("✅ Structured protocol generated successfully")

        else:
            print(f"❌ Method extraction test failed: {response.status_code}")

    except Exception as e:
        print(f"❌ Method extraction test error: {e}")


if __name__ == "__main__":
    print("🧪 Testing NanoChemGPT Method Transcription Integration")
    print("=" * 60)

    # Check if server is running
    try:
        response = requests.get("http://localhost:5000/health", timeout=5)
        if response.status_code == 200:
            print("✅ Server is running")
        else:
            print("❌ Server health check failed")
            exit(1)
    except Exception as e:
        print(f"❌ Server not accessible: {e}")
        print("Please start the NanoChemGPT server first with: python app.py")
        exit(1)

    test_transcribe_endpoint()
    test_method_paragraph_extraction()

    print("\n" + "=" * 60)
    print("🎉 Testing completed!")
