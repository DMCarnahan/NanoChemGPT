import json
import os

from app_utils import jobs as jobs_mod

import app as app_module


def test_job_status_endpoint():
    jid = 'testjob123'
    # Ensure clean slate
    if jobs_mod.JOBS.get(jid):
        jobs_mod.JOBS.pop(jid, None)

    # Set a job and verify /status/<jid> returns it
    jobs_mod.set_job(jid, status='queued', progress=0, stage='queued_harvest')

    client = app_module.app.test_client()
    resp = client.get(f"/status/{jid}")
    assert resp.status_code == 200
    data = resp.get_json()
    assert data.get('status') == 'queued'
    assert data.get('progress') == 0
    assert data.get('stage') == 'queued_harvest'

    # Mark done and verify status updates
    jobs_mod.mark_done(jid)
    resp2 = client.get(f"/status/{jid}")
    assert resp2.status_code == 200
    d2 = resp2.get_json()
    assert d2.get('status') == 'done'
    assert d2.get('progress') == 100
