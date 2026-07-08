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


def test_post_like_flow(client):
    user = signup_and_login(client, "postuser", "postuser@example.com")
    resp = client.post(
        f"/athletes/{user['id']}/posts",
        headers=user["headers"],
        json={"text": "Hello, this is a test post!"},
    )
    assert resp.status_code == 200
    assert resp.json()["success"] is True
    post_id = resp.json()["post"]["id"]

    resp = client.get("/athlete/feed", headers=user["headers"])
    assert resp.status_code == 200
    feed_items = resp.json()["items"]
    post_item = next(item for item in feed_items if item["id"] == post_id)
    assert post_item["post_data"]["is_liked"] is False
    assert post_item["post_data"]["like_count"] == 0

    resp = client.post(f"/posts/{post_id}/likes", headers=user["headers"])
    assert resp.status_code == 200
    assert resp.json()["success"] is False
    assert "cannot like your own" in resp.json()["error_message"].lower()

    other = signup_and_login(client, "postliker", "postliker@example.com")
    resp = client.post(f"/posts/{post_id}/likes", headers=other["headers"])
    assert resp.status_code == 200
    assert resp.json()["success"] is True

    resp = client.get("/athlete/feed", headers=user["headers"])
    assert resp.status_code == 200
    feed_items = resp.json()["items"]
    post_item = next(item for item in feed_items if item["id"] == post_id)
    assert post_item["post_data"]["is_liked"] is False
    assert post_item["post_data"]["like_count"] == 1

    resp = client.delete(f"/posts/{post_id}/likes", headers=other["headers"])
    assert resp.status_code == 200
    assert resp.json()["success"] is True

    resp = client.get("/athlete/feed", headers=user["headers"])
    assert resp.status_code == 200
    feed_items = resp.json()["items"]
    post_item = next(item for item in feed_items if item["id"] == post_id)
    assert post_item["post_data"]["is_liked"] is False
    assert post_item["post_data"]["like_count"] == 0


def test_cannot_like_own_activity(client):
    from tests.helpers import create_manual_activity

    user = signup_and_login(client, "selflikeact", "selflikeact@example.com")
    act_id = create_manual_activity(client, user["headers"], "Solo Run")
    resp = client.post(f"/activities/{act_id}/likes", headers=user["headers"])
    assert resp.status_code == 200
    assert resp.json()["success"] is False
    assert "cannot like your own" in resp.json()["error_message"].lower()


def test_cannot_like_own_comment(client):
    from tests.helpers import create_manual_activity

    owner = signup_and_login(client, "commentowner", "commentowner@example.com")
    other = signup_and_login(client, "commentliker", "commentliker@example.com")
    act_id = create_manual_activity(client, owner["headers"], "Group Run")
    comment_resp = client.post(
        f"/activities/{act_id}/comments",
        headers=other["headers"],
        params={"text": "Nice run"},
    )
    comment_id = comment_resp.json()["comment"]["id"]
    resp = client.post(f"/comments/{comment_id}/likes", headers=other["headers"])
    assert resp.status_code == 200
    assert resp.json()["success"] is False
    assert "cannot like your own" in resp.json()["error_message"].lower()
