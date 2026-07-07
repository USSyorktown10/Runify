"""Seed demo data for manual API exploration."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))



from app.core.security import hash_password
from app.db.session import SessionLocal
from app.models.athlete import Athlete, AthletePreferences, AthleteStats


def seed():
    db = SessionLocal()
    try:
        existing = db.query(Athlete).filter_by(username="demo").first()
        if existing:
            print("Demo user already exists (username: demo, password: demo1234)")
            return

        athlete = Athlete(
            username="demo",
            email="demo@example.com",
            password_hash=hash_password("demo1234"),
            first_name="Demo",
            last_name="Runner",
            city="San Francisco",
            state="CA",
            country="USA",
            email_verified=True,
        )
        db.add(athlete)
        db.flush()
        db.add(AthletePreferences(athlete_id=athlete.id))
        db.add(AthleteStats(athlete_id=athlete.id, threshold_pace=3.8))
        db.commit()
        print("Created demo user: username=demo, password=demo1234")
    finally:
        db.close()


if __name__ == "__main__":
    seed()
