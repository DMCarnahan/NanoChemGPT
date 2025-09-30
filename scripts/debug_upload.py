#!/usr/bin/env python3
"""
Debug script to test vector store operations
"""

import os
import sys
from pathlib import Path

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

def test_vector_store():
    print("=== Vector Store Debug ===")
    
    try:
        print("1. Testing vector store import...")
        import vector_store as vs
        print("   ✓ vector_store imported successfully")
        
        print("2. Testing add_to_store...")
        test_text = "This is a test document for debugging the upload issue."
        vs.add_to_store(test_text, tag="debug:test")
        print("   ✓ add_to_store completed successfully")
        
        print("3. Testing search...")
        results = vs.search("test document", k=3)
        print(f"   ✓ search completed, got {len(results)} chars of results")
        
        print("4. Testing UploadsVectorSearch...")
        from vector_store.uploads_vector import UploadsVectorSearch
        print("   ✓ UploadsVectorSearch imported successfully")
        
        # Test creating UploadsVectorSearch
        uploads_dir = Path("/mnt/data/uploads")
        uploads_dir.mkdir(parents=True, exist_ok=True)
        
        uvs = UploadsVectorSearch.from_folder(uploads_dir, device="cpu", max_docs=10)
        print("   ✓ UploadsVectorSearch created successfully")
        
    except Exception as e:
        print(f"   ✗ Error: {e}")
        import traceback
        traceback.print_exc()

def test_pdf_processing():
    print("\n=== PDF Processing Debug ===")
    
    try:
        print("1. Testing PDF reader imports...")
        try:
            from pypdf import PdfReader
            print("   ✓ pypdf available")
        except ImportError:
            try:
                from PyPDF2 import PdfReader
                print("   ✓ PyPDF2 available")
            except ImportError:
                print("   ✗ No PDF reader available")
                return
        
        print("2. Testing sample PDF processing...")
        # This would test with an actual PDF if available
        print("   (Skipping actual PDF test - would need sample file)")
        
    except Exception as e:
        print(f"   ✗ Error: {e}")
        import traceback
        traceback.print_exc()

def check_job_status(job_id):
    print(f"\n=== Job Status Check: {job_id} ===")
    
    try:
        import json
        job_dir = Path(os.environ.get('JOB_DIR', '/mnt/data/jobs'))
        job_file = job_dir / f"{job_id}.json"
        
        if job_file.exists():
            job_data = json.loads(job_file.read_text())
            print("Job data:")
            for key, value in job_data.items():
                print(f"  {key}: {value}")
        else:
            print(f"   Job file not found: {job_file}")
            
    except Exception as e:
        print(f"   ✗ Error checking job: {e}")

if __name__ == "__main__":
    test_vector_store()
    test_pdf_processing()
    
    # Check the specific stuck job
    if len(sys.argv) > 1:
        check_job_status(sys.argv[1])
    else:
        check_job_status("d280514f46e5c288")