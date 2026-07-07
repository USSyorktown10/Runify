"""Runify R1 API unit tests."""
from tests.helpers import signup_and_login


def test_health(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


def test_signup_and_login(client):
    user = signup_and_login(client, "testuser", "test@example.com", password="pass1234")
    assert user["token"] is not None


def test_get_me(client):
    user = signup_and_login(client, "runner1", "runner1@example.com")
    resp = client.get("/athlete/me", headers=user["headers"])
    assert resp.status_code == 200
    assert resp.json()["username"] == "runner1"


def test_update_profile(client):
    user = signup_and_login(client, "runner1b", "runner1b@example.com")
    resp = client.patch(
        "/athlete/profile",
        headers=user["headers"],
        json={"first_name": "Alex", "last_name": "Runner", "city": "Boston"},
    )
    assert resp.status_code == 200
    assert resp.json()["first_name"] == "Alex"
    assert resp.json()["city"] == "Boston"


def test_create_manual_activity(client):
    user = signup_and_login(client, "actuser", "actuser@example.com")
    resp = client.post(
        "/activities",
        headers=user["headers"],
        json={
            "name": "Morning Run",
            "activity_type": "run",
            "start_date": "2026-07-07T08:00:00+00:00",
            "elapsed_time": 3600,
            "distance": 10000,
            "metrics": [
                {
                    "key": "avg_hr",
                    "value": 150,
                    "source": "raw",
                    "unit": "bpm",
                    "display_name": "Average Heart Rate",
                }
            ],
        },
    )
    assert resp.status_code == 200
    assert resp.json()["success"] is True
    activity_id = resp.json()["id"]

    resp = client.get(f"/activities/{activity_id}", headers=user["headers"])
    assert resp.status_code == 200
    assert resp.json()["name"] == "Morning Run"
    assert resp.json()["activity_type"] == "run"


def test_gear_crud(client):
    user = signup_and_login(client, "gearuser", "gearuser@example.com")
    resp = client.post(
        "/gear",
        headers=user["headers"],
        json={"name": "Nike Pegasus", "brand_name": "Nike", "model_name": "Pegasus 41", "max_mileage": 800000},
    )
    assert resp.status_code == 200
    resp.json()["gear"]["id"]

    resp = client.get("/gear", headers=user["headers"])
    assert len(resp.json()) == 1
    assert resp.json()[0]["name"] == "Nike Pegasus"


def test_preferences(client):
    user = signup_and_login(client, "prefsuser", "prefsuser@example.com")
    resp = client.get("/preferences", headers=user["headers"])
    assert resp.status_code == 200
    assert resp.json()["measurement_system"] == "metric"

    resp = client.patch("/preferences", headers=user["headers"], json={"theme": "dark"})
    assert resp.json()["success"] is True


def test_follow_flow(client):
    user1 = signup_and_login(client, "follower1", "follower1@example.com")
    user2 = signup_and_login(client, "followee1", "followee1@example.com")
    resp = client.post(f"/athletes/{user2['id']}/follow", headers=user1["headers"])
    assert resp.status_code == 200


def test_create_club(client):
    user = signup_and_login(client, "clubuser", "clubuser@example.com")
    resp = client.post(
        "/clubs",
        headers=user["headers"],
        json={"name": "Boston Runners", "description": "Local running club", "tags": ["running"]},
    )
    assert resp.status_code == 200
    assert resp.json()["success"] is True
    assert resp.json()["club"]["name"] == "Boston Runners"
