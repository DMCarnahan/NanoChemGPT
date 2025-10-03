"""
Test Suite for NanoChemGPT

This module provides comprehensive testing for the NanoChemGPT application,
including unit tests, integration tests, and performance benchmarks.
"""

import sys
import time
from pathlib import Path

import pytest
import requests

# Add the parent directory to the path so we can import the app
sys.path.insert(0, str(Path(__file__).parent.parent))


class TestConfig:
    """Test configuration and utilities."""

    BASE_URL = "http://localhost:5000"
    TIMEOUT = 30
    TEST_DATA_DIR = Path(__file__).parent / "test_data"

    @classmethod
    def setup_test_data(cls):
        """Create test data directory and sample files."""
        cls.TEST_DATA_DIR.mkdir(exist_ok=True)

        # Sample protocol text
        sample_protocol = """
        Gold Nanoparticle Synthesis Protocol
        
        Materials:
        - HAuCl₄·3H₂O (0.5 mM, 100 mL)
        - Sodium citrate (38.8 mM, 10 mL)
        
        Procedure:
        1. Heat 100 mL of 0.5 mM HAuCl₄ solution to boiling (100°C)
        2. Add 10 mL of 38.8 mM sodium citrate solution rapidly
        3. Continue boiling for 15 minutes while stirring
        4. Cool to room temperature
        """

        sample_file = cls.TEST_DATA_DIR / "sample_protocol.txt"
        sample_file.write_text(sample_protocol)

        return sample_file


@pytest.fixture(scope="session")
def test_server():
    """Ensure test server is running."""
    try:
        response = requests.get(f"{TestConfig.BASE_URL}/health", timeout=5)
        if response.status_code == 200:
            return TestConfig.BASE_URL
    except requests.exceptions.RequestException:
        pass

    pytest.skip(
        "Test server not available. Start the application before running tests."
    )


@pytest.fixture
def sample_protocol():
    """Provide sample protocol file for testing."""
    return TestConfig.setup_test_data()


class TestHealthCheck:
    """Test basic health check endpoint."""

    def test_health_endpoint(self, test_server):
        """Test that health endpoint returns OK."""
        response = requests.get(f"{test_server}/health")
        assert response.status_code == 200

    def test_health_response_format(self, test_server):
        """Test health endpoint response format."""
        response = requests.get(f"{test_server}/health")
        # Health endpoint might return simple text or JSON
        assert response.status_code == 200
        assert len(response.text) > 0


class TestQuestionAnswering:
    """Test the main question answering functionality."""

    def test_basic_question(self, test_server):
        """Test basic question answering."""
        payload = {"question": "What is nanochemistry?", "mode": "reasoning"}
        response = requests.post(
            f"{test_server}/ask", json=payload, timeout=TestConfig.TIMEOUT
        )

        assert response.status_code == 200
        data = response.json()
        assert data.get("ok") is True
        assert "answer" in data
        assert len(data["answer"]) > 0

    def test_synthesis_protocol_question(self, test_server):
        """Test synthesis protocol generation."""
        payload = {
            "question": "How to synthesize gold nanoparticles?",
            "mode": "protocol",
            "k_doc": 3,
        }
        response = requests.post(
            f"{test_server}/ask", json=payload, timeout=TestConfig.TIMEOUT
        )

        assert response.status_code == 200
        data = response.json()
        assert data.get("ok") is True
        assert "answer" in data

        # Check for protocol-like content
        answer = data["answer"].lower()
        protocol_keywords = ["synthesis", "temperature", "solution", "heat", "add"]
        assert any(keyword in answer for keyword in protocol_keywords)

    def test_invalid_question(self, test_server):
        """Test handling of invalid or empty questions."""
        # Empty question
        payload = {"question": ""}
        response = requests.post(f"{test_server}/ask", json=payload)
        assert response.status_code == 400

        # Missing question
        payload = {"mode": "protocol"}
        response = requests.post(f"{test_server}/ask", json=payload)
        assert response.status_code == 400

    def test_question_with_parameters(self, test_server):
        """Test question answering with various parameters."""
        payload = {
            "question": "Synthesize silver nanoparticles",
            "mode": "protocol",
            "intent": "synthesis",
            "k_doc": 5,
            "k_passage": 10,
            "want_inline": True,
        }
        response = requests.post(
            f"{test_server}/ask", json=payload, timeout=TestConfig.TIMEOUT
        )

        assert response.status_code == 200
        data = response.json()
        assert data.get("ok") is True
        assert "answer" in data


class TestFileUpload:
    """Test file upload and processing functionality."""

    def test_text_file_upload(self, test_server, sample_protocol):
        """Test uploading and analyzing a text file."""
        with open(sample_protocol, "rb") as f:
            files = {"file": ("sample_protocol.txt", f, "text/plain")}
            data = {"question": "Analyze this synthesis protocol", "mode": "reasoning"}

            response = requests.post(
                f"{test_server}/ask", data=data, files=files, timeout=TestConfig.TIMEOUT
            )

        assert response.status_code == 200
        result = response.json()
        assert result.get("ok") is True
        assert "answer" in result

    def test_verbatim_extraction(self, test_server, sample_protocol):
        """Test verbatim text extraction from uploaded files."""
        with open(sample_protocol, "rb") as f:
            files = {"file": ("sample_protocol.txt", f, "text/plain")}
            data = {"question": "Quote the procedure verbatim", "mode": "protocol"}

            response = requests.post(
                f"{test_server}/ask", data=data, files=files, timeout=TestConfig.TIMEOUT
            )

        assert response.status_code == 200
        result = response.json()
        assert result.get("ok") is True

        # Check if verbatim mode was triggered
        if "verbatim" in result.get("rationale", "").lower():
            assert "attachment" in result["answer"].lower()


class TestProtocolConversion:
    """Test protocol text to robot operations conversion."""

    def test_basic_conversion(self, test_server):
        """Test basic protocol text conversion."""
        payload = {
            "text": "Heat the solution to 80°C and stir for 2 hours",
            "validate": True,
        }
        response = requests.post(
            f"{test_server}/convert", json=payload, timeout=TestConfig.TIMEOUT
        )

        assert response.status_code == 200
        data = response.json()
        assert data.get("ok") is True
        assert "operations" in data
        assert len(data["operations"]) > 0

        # Check for expected operations
        operations = data["operations"]
        operation_types = [op.get("action") for op in operations]
        assert "heat" in operation_types

    def test_complex_protocol_conversion(self, test_server):
        """Test conversion of complex protocol with multiple steps."""
        protocol_text = """
        Heat the gold chloride solution to 100°C while stirring at 300 rpm.
        Add 10 mL of sodium citrate dropwise over 2 minutes.
        Continue heating for 15 minutes.
        Cool to room temperature and centrifuge at 8000 rpm for 10 minutes.
        """

        payload = {
            "text": protocol_text,
            "target_ops": ["heat", "mix", "add", "wait", "cool", "centrifuge"],
            "validate": True,
        }
        response = requests.post(
            f"{test_server}/convert", json=payload, timeout=TestConfig.TIMEOUT
        )

        assert response.status_code == 200
        data = response.json()
        assert data.get("ok") is True

        operations = data["operations"]
        assert len(operations) >= 3  # Should have multiple operations

        # Check validation results
        if "validation" in data:
            validation = data["validation"]
            assert "valid" in validation

    def test_invalid_protocol_text(self, test_server):
        """Test handling of invalid protocol text."""
        # Empty text
        payload = {"text": ""}
        response = requests.post(f"{test_server}/convert", json=payload)
        assert response.status_code == 400

        # Non-protocol text
        payload = {"text": "This is not a protocol", "validate": True}
        response = requests.post(f"{test_server}/convert", json=payload)
        # Should still return 200 but with empty or minimal operations
        assert response.status_code == 200


class TestUtilityFunctions:
    """Test utility functions and helper methods."""

    def test_text_sanitization(self):
        """Test the _s() text sanitization function."""
        from app import _s

        # Test basic cases
        assert _s(None) == ""
        assert _s("  hello  ") == "hello"
        assert _s(123) == "123"
        assert _s("") == ""

    def test_safe_id_conversion(self):
        """Test the _safe_id() ObjectId conversion function."""
        from app import _safe_id

        # Test with invalid input
        assert _safe_id("invalid") is None
        assert _safe_id(None) is None
        assert _safe_id("") is None

    def test_key_stringification(self):
        """Test the _stringify_keys() function."""
        from app import _stringify_keys

        # Test basic dict
        test_dict = {1: "one", 2: "two"}
        result = _stringify_keys(test_dict)
        assert result == {"1": "one", "2": "two"}

        # Test nested structure
        nested = {1: {"a": "value"}, 2: [3, 4]}
        result = _stringify_keys(nested)
        assert result == {"1": {"a": "value"}, "2": [3, 4]}

    def test_text_chunking(self):
        """Test the _best_chunks_from_text() function."""
        from app import _best_chunks_from_text

        text = "Gold nanoparticles synthesis. Silver nanoparticles formation. Copper nanowires."
        query = "gold synthesis"

        chunks = _best_chunks_from_text(text, query, max_chunk_chars=50, top_k=2)
        assert len(chunks) <= 2
        assert len(chunks) > 0

        # First chunk should contain gold-related content
        assert "gold" in chunks[0].lower() or "synthesis" in chunks[0].lower()


class TestPerformance:
    """Test system performance and response times."""

    def test_response_time(self, test_server):
        """Test that responses come back within reasonable time."""
        start_time = time.time()

        payload = {"question": "What is gold?", "mode": "reasoning"}
        response = requests.post(
            f"{test_server}/ask", json=payload, timeout=TestConfig.TIMEOUT
        )

        end_time = time.time()
        response_time = end_time - start_time

        assert response.status_code == 200
        assert response_time < 30  # Should respond within 30 seconds

    def test_concurrent_requests(self, test_server):
        """Test handling of multiple concurrent requests."""
        import concurrent.futures

        def make_request():
            payload = {"question": "Test question", "mode": "reasoning"}
            response = requests.post(f"{test_server}/ask", json=payload, timeout=15)
            return response.status_code == 200

        # Run 3 concurrent requests
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
            futures = [executor.submit(make_request) for _ in range(3)]
            results = [
                future.result() for future in concurrent.futures.as_completed(futures)
            ]

        # At least 2 out of 3 should succeed (allows for some load issues)
        assert sum(results) >= 2


class TestErrorHandling:
    """Test error handling and edge cases."""

    def test_malformed_json(self, test_server):
        """Test handling of malformed JSON requests."""
        response = requests.post(
            f"{test_server}/ask",
            data="invalid json",
            headers={"Content-Type": "application/json"},
        )
        # Should return 400 for malformed JSON
        assert response.status_code in [400, 422]

    def test_missing_required_fields(self, test_server):
        """Test handling of requests with missing required fields."""
        # Missing question field
        payload = {"mode": "protocol"}
        response = requests.post(f"{test_server}/ask", json=payload)
        assert response.status_code == 400

    def test_invalid_mode(self, test_server):
        """Test handling of invalid mode values."""
        payload = {"question": "Test question", "mode": "invalid_mode"}
        response = requests.post(f"{test_server}/ask", json=payload)
        # Should either accept it or return 400, but not crash
        assert response.status_code in [200, 400]

    def test_very_long_question(self, test_server):
        """Test handling of extremely long questions."""
        long_question = "What is nanochemistry? " * 1000  # Very long question
        payload = {"question": long_question, "mode": "reasoning"}
        response = requests.post(f"{test_server}/ask", json=payload, timeout=45)
        # Should handle gracefully, either process or reject
        assert response.status_code in [200, 400, 413]


class TestIntegration:
    """Integration tests that test multiple components together."""

    def test_end_to_end_workflow(self, test_server, sample_protocol):
        """Test complete workflow from upload to protocol conversion."""
        # Step 1: Upload and analyze file
        with open(sample_protocol, "rb") as f:
            files = {"file": ("sample_protocol.txt", f, "text/plain")}
            data = {
                "question": "Analyze this protocol and provide step-by-step instructions",
                "mode": "protocol",
            }

            response1 = requests.post(
                f"{test_server}/ask", data=data, files=files, timeout=TestConfig.TIMEOUT
            )

        assert response1.status_code == 200
        result1 = response1.json()
        assert result1.get("ok") is True

        # Step 2: Convert the protocol to robot operations
        protocol_text = result1["answer"]
        conversion_payload = {"text": protocol_text, "validate": True}

        response2 = requests.post(
            f"{test_server}/convert",
            json=conversion_payload,
            timeout=TestConfig.TIMEOUT,
        )

        assert response2.status_code == 200
        result2 = response2.json()
        assert result2.get("ok") is True

        # Should have generated some operations
        if "operations" in result2:
            assert len(result2["operations"]) > 0


if __name__ == "__main__":
    # Run tests when executed directly
    pytest.main([__file__, "-v"])
