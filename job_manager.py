#!/usr/bin/env python3
"""
Utility to manually complete stuck upload jobs
"""

import json
import os
import time
from pathlib import Path

def list_jobs():
    """List all current jobs"""
    job_dir = Path("data/jobs")
    if not job_dir.exists():
        print("No jobs directory found")
        return
    
    job_files = list(job_dir.glob("*.json"))
    if not job_files:
        print("No job files found")
        return
    
    print(f"Found {len(job_files)} job files:")
    for job_file in job_files:
        try:
            job_data = json.loads(job_file.read_text())
            status = job_data.get('status', 'unknown')
            progress = job_data.get('progress', 0)
            filename = job_data.get('filename', 'unknown')
            created_at = job_data.get('created_at', 0)
            updated_at = job_data.get('updated_at', 0)
            
            # Calculate age
            current_time = time.time()
            age_seconds = current_time - created_at if created_at else 0
            
            print(f"  {job_file.stem}: {status} ({progress}%) - {filename}")
            print(f"    Age: {age_seconds:.0f}s, Last update: {current_time - updated_at if updated_at else 0:.0f}s ago")
            
        except Exception as e:
            print(f"  {job_file.stem}: Error reading - {e}")

def complete_job(job_id):
    """Force complete a specific job"""
    job_dir = Path("data/jobs")
    job_file = job_dir / f"{job_id}.json"
    
    if not job_file.exists():
        print(f"Job file not found: {job_file}")
        return False
    
    try:
        job_data = json.loads(job_file.read_text())
        
        if job_data.get('status') == 'done':
            print(f"Job {job_id} is already completed")
            return True
        
        # Force complete the job
        job_data['status'] = 'done'
        job_data['updated_at'] = time.time()
        job_data['warning'] = 'Manually force-completed'
        
        job_file.write_text(json.dumps(job_data, indent=2))
        print(f"Job {job_id} force-completed successfully")
        return True
        
    except Exception as e:
        print(f"Error completing job {job_id}: {e}")
        return False

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        command = sys.argv[1]
        if command == "list":
            list_jobs()
        elif command == "complete" and len(sys.argv) > 2:
            job_id = sys.argv[2]
            complete_job(job_id)
        else:
            print("Usage:")
            print("  python job_manager.py list")
            print("  python job_manager.py complete <job_id>")
    else:
        print("Current jobs:")
        list_jobs()
        print("\nTo complete a stuck job:")
        print("  python job_manager.py complete <job_id>")