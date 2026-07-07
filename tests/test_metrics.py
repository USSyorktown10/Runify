"""Test metrics engine with sample GPX data."""
import os
import tempfile

import pytest

from app.core.security import hash_password
from app.models.activity import Upload
from app.models.athlete import Athlete, AthletePreferences, AthleteStats
from app.services.metrics_engine.pipeline import process_upload
from tests.helpers import GPX_SAMPLE


@pytest.fixture
def metrics_db(db):
    athlete = Athlete(
        username="metrics_user",
        email="metrics@example.com",
        password_hash=hash_password("test"),
    )
    db.add(athlete)
    db.flush()
    db.add(AthletePreferences(athlete_id=athlete.id))
    db.add(AthleteStats(athlete_id=athlete.id, threshold_pace=3.5))
    db.commit()
    return db


def test_gpx_pipeline(metrics_db):
    with tempfile.NamedTemporaryFile(suffix=".gpx", delete=False, mode="w") as f:
        f.write(GPX_SAMPLE)
        path = f.name

    upload = Upload(athlete_id=metrics_db.query(Athlete).first().id, file_path=path, file_name="test.gpx")
    metrics_db.add(upload)
    metrics_db.flush()

    activity = process_upload(metrics_db, upload, threshold_pace=3.5)
    metrics_db.commit()

    assert activity.distance > 0
    assert activity.moving_time > 0
    assert len(activity.metrics) > 0
    assert any(m.key == "distance" for m in activity.metrics)
    assert len(activity.streams) > 0

    os.unlink(path)
