"""Test helpers for creating users and API resources."""
from __future__ import annotations

import httpx

GPX_SAMPLE = """<?xml version="1.0" encoding="UTF-8"?>
<gpx version="1.1" creator="Runify Test">
  <trk>
    <trkseg>
      <trkpt lat="42.3601" lon="-71.0589"><time>2026-07-01T08:00:00Z</time><ele>10</ele></trkpt>
      <trkpt lat="42.3610" lon="-71.0570"><time>2026-07-01T08:01:00Z</time><ele>12</ele></trkpt>
      <trkpt lat="42.3620" lon="-71.0550"><time>2026-07-01T08:02:00Z</time><ele>15</ele></trkpt>
      <trkpt lat="42.3630" lon="-71.0530"><time>2026-07-01T08:03:00Z</time><ele>18</ele></trkpt>
      <trkpt lat="42.3640" lon="-71.0510"><time>2026-07-01T08:04:00Z</time><ele>20</ele></trkpt>
      <trkpt lat="42.3650" lon="-71.0490"><time>2026-07-01T08:05:00Z</time><ele>22</ele></trkpt>
      <trkpt lat="42.3660" lon="-71.0470"><time>2026-07-01T08:06:00Z</time><ele>25</ele></trkpt>
      <trkpt lat="42.3670" lon="-71.0450"><time>2026-07-01T08:07:00Z</time><ele>28</ele></trkpt>
      <trkpt lat="42.3680" lon="-71.0430"><time>2026-07-01T08:08:00Z</time><ele>30</ele></trkpt>
      <trkpt lat="42.3690" lon="-71.0410"><time>2026-07-01T08:09:00Z</time><ele>32</ele></trkpt>
    </trkseg>
  </trk>
</gpx>
"""


def signup_and_login(
    client: httpx.Client,
    username: str,
    email: str,
    password: str = "securepass123",
    metadata: dict | None = None,
) -> dict:
    payload: dict = {"username": username, "email": email, "password": password}
    if metadata:
        payload["metadata"] = metadata
    resp = client.post("/authentication/signup", json=payload)
    assert resp.status_code == 200, resp.text
    assert resp.json()["success"] is True

    resp = client.post("/authentication/login", json={"username": username, "password": password})
    assert resp.status_code == 200, resp.text
    token = resp.json()["session_token"]
    assert token is not None

    me = client.get("/athlete/me", headers={"Authorization": f"Bearer {token}"}).json()
    return {
        "username": username,
        "email": email,
        "token": token,
        "headers": {"Authorization": f"Bearer {token}"},
        "id": me["id"],
        "me": me,
    }


def create_manual_activity(
    client: httpx.Client,
    headers: dict,
    name: str,
    distance: float = 10000,
    elapsed_time: int = 3600,
    activity_type: str = "run",
    start_date: str = "2026-07-07T08:00:00+00:00",
) -> str:
    resp = client.post(
        "/activities",
        headers=headers,
        json={
            "name": name,
            "activity_type": activity_type,
            "start_date": start_date,
            "elapsed_time": elapsed_time,
            "distance": distance,
            "description": f"Test activity: {name}",
            "metrics": [
                {
                    "key": "avg_hr",
                    "value": 145,
                    "source": "raw",
                    "unit": "bpm",
                    "display_name": "Average Heart Rate",
                }
            ],
        },
    )
    assert resp.status_code == 200, resp.text
    assert resp.json()["success"] is True
    return resp.json()["id"]


def upload_gpx_activity(client: httpx.Client, headers: dict, filename: str = "run.gpx") -> str:
    """Upload GPX and wait for processing (polls since background task runs in uvicorn container)."""
    resp = client.post(
        "/upload",
        headers=headers,
        files={"file": (filename, GPX_SAMPLE.encode(), "application/gpx+xml")},
        data={"metadata": "{}"},
    )
    assert resp.status_code == 200, resp.text
    upload_id = resp.json()["upload_id"]

    import time
    for _ in range(100):  # Wait up to 10 seconds (100 * 0.1s)
        status_resp = client.get(f"/upload/{upload_id}", headers=headers)
        assert status_resp.status_code == 200, status_resp.text
        data = status_resp.json()
        if data["status"] == "completed":
            assert data["activity_id"] is not None
            return data["activity_id"]
        elif data["status"] == "failed":
            raise Exception(f"Upload processing failed: {data.get('error_message')}")
        time.sleep(0.1)

    raise TimeoutError(f"Upload processing timed out for upload {upload_id}")
