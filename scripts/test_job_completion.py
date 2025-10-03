#!/usr/bin/env python3
"""
Test script to verify upload job auto-completion
"""

import time

import requests


def test_job_completion(job_id):
    """Test the auto-completion mechanism for a stuck job"""
    print(f"Testing job completion for: {job_id}")

    base_url = "http://localhost:8000"
    status_url = f"{base_url}/status/{job_id}"

    try:
        # Check initial status
        print("1. Checking initial status...")
        response = requests.get(status_url)
        if response.status_code == 200:
            job_data = response.json()
            print(f"   Status: {job_data.get('status')}")
            print(f"   Progress: {job_data.get('progress')}")
            print(f"   Stage: {job_data.get('stage', 'N/A')}")
            print(f"   Updated at: {job_data.get('updated_at')}")

            if job_data.get("status") == "processing":
                print("\n2. Job is processing, waiting for auto-completion...")

                # Poll status for up to 45 seconds
                for i in range(45):
                    time.sleep(1)
                    response = requests.get(status_url)
                    if response.status_code == 200:
                        job_data = response.json()
                        status = job_data.get("status")
                        print(f"   [{i+1}s] Status: {status}")

                        if status == "done":
                            print(
                                f"   ✅ Job completed! Warning: {job_data.get('warning', 'None')}"
                            )
                            break
                        elif status == "error":
                            print(
                                f"   ❌ Job failed: {job_data.get('error', 'Unknown error')}"
                            )
                            break
                    else:
                        print(f"   Status check failed: {response.status_code}")
                else:
                    print("   ⏰ Job didn't complete within 45 seconds")
            else:
                print(f"   Job status is already: {job_data.get('status')}")

        elif response.status_code == 404:
            print("   Job not found")
        else:
            print(f"   Status check failed: {response.status_code}")

    except requests.ConnectionError:
        print("   ❌ Cannot connect to server. Is the app running?")
    except Exception as e:
        print(f"   ❌ Error: {e}")


if __name__ == "__main__":
    # Test with the current stuck job
    test_job_completion("0597ae6a374af86a")
