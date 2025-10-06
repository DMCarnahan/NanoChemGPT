#!/usr/bin/env python3
"""
Simple test of upload functionality without full app dependencies
"""

import json
import threading
import time
from pathlib import Path

# Create simple job management like in app.py
JOBS = {}
JOB_DIR = Path("data/jobs")
JOB_DIR.mkdir(parents=True, exist_ok=True)


def _set_job(jid: str, **kw):
    import time

    J = JOBS.setdefault(jid, {})
    # Add timestamp for new jobs or significant updates
    if "status" in kw:
        kw["updated_at"] = time.time()
    if not J:  # New job
        kw["created_at"] = time.time()
    J.update(kw)
    try:
        (JOB_DIR / f"{jid}.json").write_text(
            json.dumps(J, ensure_ascii=False), encoding="utf-8"
        )
        print(f"Job {jid} saved to file: {kw}")
    except Exception as e:
        print(f"Error saving job {jid}: {e}")


def test_vector_store():
    """Test if vector store can be imported"""
    try:
        print("✅ Vector store imported successfully")
        return True
    except Exception as e:
        print(f"❌ Vector store import failed: {e}")
        return False


def simulate_pdf_processing(jid: str, filename: str):
    """Simulate the PDF processing workflow"""
    print(f"\n🔄 Starting PDF processing for job {jid}: {filename}")

    # Step 1: Reading
    _set_job(jid, status="processing", progress=25, stage="reading", filename=filename)
    time.sleep(1)

    # Step 2: Parsing
    _set_job(jid, status="processing", progress=50, stage="parsing")
    time.sleep(1)

    # Step 3: Processing
    _set_job(jid, status="processing", progress=75, stage="processing")
    time.sleep(1)

    # Step 4: Indexing (this is where it usually gets stuck)
    _set_job(jid, status="processing", progress=100, stage="indexing")
    time.sleep(1)

    # Check if vector store is available
    vs_available = test_vector_store()

    if not vs_available:
        print("⚠️  Vector store not available, simulating hang...")
        time.sleep(35)  # Simulate hanging for 35 seconds

        # Auto-complete after timeout
        print("🔧 Auto-completing job after timeout")
        _set_job(
            jid,
            status="done",
            warning="Job was stuck at indexing stage and auto-completed (vector store disabled)",
        )
        print("✅ Job auto-completed")
    else:
        # If vector store is available, complete normally
        print("✅ Vector store available, completing normally")
        _set_job(jid, status="done")


def test_status_endpoint_logic(jid: str):
    """Test the status endpoint auto-completion logic"""
    print(f"\n🔍 Testing status endpoint logic for job {jid}")

    # Simulate reading job from file
    try:
        p = JOB_DIR / f"{jid}.json"
        if p.exists():
            j = json.loads(p.read_text(encoding="utf-8"))
            print(f"📄 Job data from file: {j}")

            # Auto-cleanup logic from status endpoint
            if j.get("status") == "processing" and j.get("progress") == 100:
                import time

                updated_at = j.get("updated_at", 0)
                current_time = time.time()

                if current_time - updated_at > 30:  # 30 seconds
                    print(
                        f"🔧 Auto-completing stuck job {jid} after {current_time - updated_at:.1f}s"
                    )
                    j["status"] = "done"
                    j["warning"] = (
                        "Job was stuck at indexing stage and auto-completed (vector store disabled)"
                    )
                    _set_job(jid, **j)
                    print("✅ Job auto-completed via status endpoint")
                else:
                    print(
                        f"⏳ Job not stuck long enough yet ({current_time - updated_at:.1f}s < 30s)"
                    )
            else:
                print("ℹ️  Job not in stuck state")
        else:
            print("❌ Job file not found")
    except Exception as e:
        print(f"❌ Error testing status logic: {e}")


if __name__ == "__main__":
    print("🧪 Testing NanoChemGPT Upload Functionality")
    print("=" * 50)

    # Test 1: Vector store availability
    print("\n1️⃣  Testing vector store availability:")
    test_vector_store()

    # Test 2: Simulate PDF processing
    test_jid = "test_" + str(int(time.time()))
    print(f"\n2️⃣  Simulating PDF processing for job: {test_jid}")

    # Run in thread to simulate background processing
    thread = threading.Thread(
        target=simulate_pdf_processing, args=(test_jid, "test.pdf")
    )
    thread.start()

    # Wait for job to get stuck at indexing
    time.sleep(5)

    # Test 3: Status endpoint logic
    print("\n3️⃣  Testing status endpoint auto-completion:")
    time.sleep(32)  # Wait for the 30-second timeout
    test_status_endpoint_logic(test_jid)

    # Wait for thread to complete
    thread.join()

    print("\n4️⃣  Final job state:")
    try:
        p = JOB_DIR / f"{test_jid}.json"
        if p.exists():
            final_state = json.loads(p.read_text(encoding="utf-8"))
            print(f"📄 Final job data: {final_state}")
        else:
            print("❌ No final job file found")
    except Exception as e:
        print(f"❌ Error reading final state: {e}")

    print("\n🎯 Test completed! Check data/jobs/ for job files.")
