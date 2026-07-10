"""Populate the database with a large, realistic dataset for end-to-end testing.

Usage:
    python scripts/seed_bulk.py                  # defaults: 50 athletes, 15 activities each
    python scripts/seed_bulk.py --clear          # wipe existing data first
    python scripts/seed_bulk.py --athletes 100 --activities-per-athlete 25
    python scripts/seed_bulk.py --seed 42        # reproducible randomness
"""
from __future__ import annotations

import argparse
import math
import random
import sys
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import polyline as polyline_lib
from sqlalchemy import select, text

sys.path.insert(0, str(Path(__file__).parent.parent))

import app.models.activity  # noqa: F401 — register ORM tables
import app.models.athlete  # noqa: F401
import app.models.auth  # noqa: F401
import app.models.segment  # noqa: F401
import app.models.social  # noqa: F401
from app.core.security import hash_password
from app.db.schema_sync import sync_schema
from app.db.session import SessionLocal
from app.models.activity import (
    Activity,
    ActivityDistribution,
    ActivityLap,
    ActivityMetric,
    ActivitySplit,
    ActivityStream,
    ActivityZone,
    Gear,
)
from app.models.athlete import Athlete, AthletePreferences, AthleteStats, PersonalRecord
from app.models.segment import Route, Segment, SegmentEffort, SegmentStar
from app.models.social import (
    AthletePost,
    Club,
    ClubMember,
    ClubPost,
    Comment,
    Follow,
    Like,
    Notification,
)

DEFAULT_PASSWORD = "demo1234"

FEATURED_ATHLETES = [
    {
        "username": "demo",
        "email": "demo@runify.app",
        "first_name": "Demo",
        "last_name": "Runner",
        "city": "San Francisco",
        "state": "CA",
        "gender": "female",
        "birthdate": date(1992, 6, 15),
        "weight_kg": 58.0,
        "height_cm": 168.0,
        "threshold_pace": 3.8,
    },
    {
        "username": "mayachen",
        "email": "maya@runify.app",
        "first_name": "Maya",
        "last_name": "Chen",
        "city": "Oakland",
        "state": "CA",
        "gender": "female",
        "birthdate": date(1994, 3, 22),
        "weight_kg": 55.0,
        "threshold_pace": 3.6,
    },
    {
        "username": "jamesruns",
        "email": "james@runify.app",
        "first_name": "James",
        "last_name": "Okonkwo",
        "city": "Berkeley",
        "state": "CA",
        "gender": "male",
        "birthdate": date(1990, 11, 8),
        "weight_kg": 72.0,
        "threshold_pace": 3.4,
    },
    {
        "username": "sofiareyes",
        "email": "sofia@runify.app",
        "first_name": "Sofia",
        "last_name": "Reyes",
        "city": "San Jose",
        "state": "CA",
        "gender": "female",
        "birthdate": date(1996, 1, 30),
        "weight_kg": 52.0,
        "threshold_pace": 3.9,
    },
    {
        "username": "eli_n",
        "email": "eli@runify.app",
        "first_name": "Eli",
        "last_name": "Nakamura",
        "city": "Palo Alto",
        "state": "CA",
        "gender": "male",
        "birthdate": date(1993, 7, 12),
        "weight_kg": 65.0,
        "threshold_pace": 3.5,
    },
]

FIRST_NAMES = [
    "Alex", "Jordan", "Taylor", "Morgan", "Casey", "Riley", "Quinn", "Avery",
    "Sam", "Jamie", "Drew", "Blake", "Cameron", "Reese", "Skyler", "Parker",
    "Noah", "Emma", "Liam", "Olivia", "Ethan", "Ava", "Lucas", "Mia",
    "Marco", "Priya", "Yuki", "Amara", "Diego", "Lin", "Fatima", "Omar",
]
LAST_NAMES = [
    "Smith", "Johnson", "Williams", "Brown", "Jones", "Garcia", "Miller", "Davis",
    "Rodriguez", "Martinez", "Lee", "Kim", "Patel", "Nguyen", "Anderson", "Thomas",
    "Jackson", "White", "Harris", "Clark", "Lewis", "Walker", "Hall", "Young",
]
CITIES = [
    ("San Francisco", "CA"), ("Oakland", "CA"), ("Berkeley", "CA"), ("San Jose", "CA"),
    ("Palo Alto", "CA"), ("Marin", "CA"), ("Sacramento", "CA"), ("Portland", "OR"),
    ("Seattle", "WA"), ("Denver", "CO"), ("Austin", "TX"), ("Boston", "MA"),
    ("Chicago", "IL"), ("New York", "NY"), ("Boulder", "CO"), ("Bend", "OR"),
]
ACTIVITY_NAMES = [
    "Morning easy run", "Lunch break jog", "Tempo Tuesday", "Long run Sunday",
    "Track intervals", "Hill repeats", "Recovery shakeout", "Progression run",
    "Fartlek fun", "Trail adventure", "Park loop", "River path cruise",
    "Pre-race taper", "Post-work decompression", "Sunrise miles", "Sunset cruise",
]
RUN_TYPES = ["run", "run", "run", "trail_run", "track_run"]
DEVICES = [
    "Garmin Forerunner 265", "Garmin Fenix 7", "Apple Watch Ultra",
    "Coros Pace 3", "Polar Vantage V3", "Suunto Race", "Wahoo Elemnt",
]
SHOE_BRANDS = [
    ("Nike", "Pegasus 41"), ("Nike", "Vaporfly 3"), ("Saucony", "Endorphin Pro 4"),
    ("Brooks", "Ghost 16"), ("Hoka", "Clifton 9"), ("Asics", "Novablast 4"),
    ("New Balance", "SC Elite v4"), ("On", "Cloudmonster"),
]
CLUB_TEMPLATES = [
    ("Bay Area Run Collective", "Weekly group runs around the Bay.", ["running", "social", "bay-area"]),
    ("Early Birds Track Club", "Track workouts before sunrise.", ["track", "speed", "intervals"]),
    ("Trail Tamers", "Dirt, elevation, and post-run coffee.", ["trail", "ultra", "hills"]),
    ("Marathon Builders", "Building toward the next 26.2.", ["marathon", "long-run", "training"]),
    ("City Striders", "Urban exploring one mile at a time.", ["city", "explore", "community"]),
]
SEGMENT_TEMPLATES = [
    ("JFK Drive climb", 37.7694, -122.4862, 37.776, -122.475, 1200, 2.1),
    ("Great Highway sprint", 37.762, -122.510, 37.758, -122.498, 800, 0.2),
    ("Stow Lake loop", 37.769, -122.477, 37.771, -122.472, 2100, 1.5),
    ("Twin Peaks ascent", 37.752, -122.447, 37.754, -122.443, 900, 8.5),
    ("Embarcadero dash", 37.795, -122.393, 37.808, -122.400, 1500, 0.0),
]
COMMENT_SNIPPETS = [
    "Great pace today!", "Crushing it!", "Love this route.", "Solid effort.",
    "Those splits are 🔥", "Inspiring run.", "Need to try this loop.", "Nice work!",
    "Beast mode.", "Recovery day hero.", "Sub-4 soon!", "Beautiful morning for it.",
]
POST_SNIPPETS = [
    "I ate 500 children today!",
    "I committed 32 war crimes :)",
    "UwU",
    "I just sold $2T of SpaceX",
    "I just TKOd my grandma",
]

SF_CENTER = (37.7749, -122.4194)


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def random_start_date(rng: random.Random, days_back: int = 180) -> datetime:
    offset = rng.randint(0, days_back * 24 * 3600)
    return utc_now() - timedelta(seconds=offset)


def encode_polyline(coords: list[tuple[float, float]]) -> str:
    if not coords:
        return ""
    return polyline_lib.encode(coords)


def random_route_polyline(rng: random.Random, points: int = 40) -> tuple[str, str]:
    lat, lng = SF_CENTER
    lat += rng.uniform(-0.04, 0.04)
    lng += rng.uniform(-0.06, 0.06)
    coords: list[tuple[float, float]] = []
    for _ in range(points):
        lat += rng.uniform(-0.0015, 0.0015)
        lng += rng.uniform(-0.002, 0.002)
        coords.append((lat, lng))
    summary = coords[:: max(1, len(coords) // 20)]
    return encode_polyline(coords), encode_polyline(summary)


def pace_to_speed_mps(pace_min_per_km: float) -> float:
    if pace_min_per_km <= 0:
        return 3.0
    return 1000.0 / (pace_min_per_km * 60.0)


def build_activity_metrics(rng: random.Random, distance: float, pace: float) -> list[ActivityMetric]:
    speed = pace_to_speed_mps(pace)
    hr = rng.randint(125, 175)
    elev = rng.randint(20, 450)
    vo2 = round(rng.uniform(42, 58), 1)
    return [
        ActivityMetric(key="distance", value=distance, source="raw", unit="m", display_name="Distance"),
        ActivityMetric(key="avg_speed", value=round(speed, 3), source="calculated", unit="m/s", display_name="Avg speed"),
        ActivityMetric(key="avg_hr", value=hr, source="raw", unit="bpm", display_name="Average Heart Rate"),
        ActivityMetric(key="elevation_gain", value=elev, source="calculated", unit="m", display_name="Elevation gain"),
        ActivityMetric(key="vo2_max", value=vo2, source="calculated", unit="ml/kg/min", display_name="VO2 max"),
    ]


def build_zones(rng: random.Random, elapsed: int) -> list[ActivityZone]:
    remaining = elapsed
    zone_times = []
    for _ in range(4):
        chunk = rng.randint(0, remaining // 2) if remaining > 60 else 0
        zone_times.append(chunk)
        remaining -= chunk
    zone_times.append(max(remaining, 0))
    return [
        ActivityZone(
            key="heart_rate",
            display_name="Heart rate zones",
            unit="bpm",
            reference_value=190.0,
            reference_name="Max HR",
            zones=[
                {"zone_index": 1, "min_value": 100, "max_value": 120, "time_in_seconds": zone_times[0]},
                {"zone_index": 2, "min_value": 120, "max_value": 140, "time_in_seconds": zone_times[1]},
                {"zone_index": 3, "min_value": 140, "max_value": 160, "time_in_seconds": zone_times[2]},
                {"zone_index": 4, "min_value": 160, "max_value": 175, "time_in_seconds": zone_times[3]},
            ],
        )
    ]


def build_distributions(rng: random.Random, pace: float, elapsed: int) -> list[ActivityDistribution]:
  fast = rng.randint(int(elapsed * 0.1), int(elapsed * 0.3))
  mid = rng.randint(int(elapsed * 0.3), int(elapsed * 0.5))
  slow = max(elapsed - fast - mid, 0)
  return [
      ActivityDistribution(
          key="pace",
          display_name="Pace distribution",
          unit="min/km",
          buckets=[
              {"min_value": round(pace - 0.5, 2), "max_value": pace, "time_in_seconds": fast},
              {"min_value": pace, "max_value": round(pace + 0.5, 2), "time_in_seconds": mid},
              {"min_value": round(pace + 0.5, 2), "max_value": round(pace + 1.5, 2), "time_in_seconds": slow},
          ],
      )
  ]


def build_streams(rng: random.Random, elapsed: int, points: int = 60) -> list[ActivityStream]:
    step = max(elapsed // points, 1)
    times = list(range(0, elapsed, step))
    if not times:
        times = [0]
    base_hr = rng.randint(130, 150)
    hr_data = [base_hr + rng.randint(-15, 25) for _ in times]
    alt_data = [rng.randint(5, 120) + int(30 * math.sin(i / 8)) for i in range(len(times))]
    return [
        ActivityStream(
            metric_key="time",
            stream_type="raw",
            axis_type="time",
            resolution="high",
            original_size=len(times),
            data=times,
            axis=times,
        ),
        ActivityStream(
            metric_key="heartrate",
            stream_type="raw",
            axis_type="time",
            resolution="high",
            original_size=len(hr_data),
            data=hr_data,
            axis=times,
        ),
        ActivityStream(
            metric_key="altitude",
            stream_type="raw",
            axis_type="distance",
            resolution="high",
            original_size=len(alt_data),
            data=alt_data,
            axis=[i * 200 for i in range(len(alt_data))],
        ),
    ]


def build_splits(distance: float, elapsed: int, pace: float) -> list[ActivitySplit]:
    speed = pace_to_speed_mps(pace)
    splits: list[ActivitySplit] = []
    km = int(distance // 1000)
    for i in range(km):
        splits.append(
            ActivitySplit(
                index=i + 1,
                distance=1000.0,
                elapsed_time=int(elapsed / max(km, 1)),
                elevation_difference=random.uniform(-5, 15),
                average_speed=speed * random.uniform(0.95, 1.05),
            )
        )
    return splits


def build_laps(rng: random.Random, start: datetime, distance: float, elapsed: int) -> list[ActivityLap]:
    lap_count = rng.randint(2, 5)
    laps: list[ActivityLap] = []
    remaining_dist = distance
    remaining_time = elapsed
    lap_start = start
    for i in range(lap_count):
        is_last = i == lap_count - 1
        lap_dist = remaining_dist if is_last else remaining_dist / (lap_count - i)
        lap_time = remaining_time if is_last else remaining_time // (lap_count - i)
        speed = lap_dist / lap_time if lap_time else 3.0
        laps.append(
            ActivityLap(
                lap_index=i + 1,
                name=f"Lap {i + 1}",
                start_date=lap_start,
                elapsed_time=lap_time,
                moving_time=lap_time,
                distance=lap_dist,
                average_speed=speed,
            )
        )
        lap_start += timedelta(seconds=lap_time)
        remaining_dist -= lap_dist
        remaining_time -= lap_time
    return laps


def ensure_schema(db) -> None:
    """Create missing tables and add columns introduced after initial deploy."""
    sync_schema()


def clear_database(db) -> None:
    tables = [
        "email_outbox", "auth_tokens", "auth_sessions",
        "route_waypoints", "routes",
        "segment_efforts", "segment_stars", "segments",
        "reports", "integrations", "notifications",
        "club_posts", "club_join_requests", "club_invites", "club_members", "clubs",
        "likes", "comments", "athlete_posts",
        "blocks", "follows",
        "activity_splits", "activity_laps", "activity_streams",
        "activity_zones", "activity_distributions", "activity_metrics",
        "uploads", "activities", "gear",
        "personal_records", "athlete_streams", "athlete_stats", "athlete_preferences", "athletes",
    ]
    db.execute(text(f"TRUNCATE TABLE {', '.join(tables)} RESTART IDENTITY CASCADE"))
    db.commit()
    print("Cleared all tables.")


def create_athlete(db, rng: random.Random, spec: dict) -> Athlete:
    athlete = Athlete(
        username=spec["username"],
        email=spec["email"],
        password_hash=hash_password(spec.get("password", DEFAULT_PASSWORD)),
        first_name=spec["first_name"],
        last_name=spec["last_name"],
        city=spec.get("city", ""),
        state=spec.get("state", ""),
        country=spec.get("country", "USA"),
        gender=spec.get("gender"),
        birthdate=spec.get("birthdate"),
        weight_kg=spec.get("weight_kg"),
        height_cm=spec.get("height_cm"),
        email_verified=True,
    )
    db.add(athlete)
    db.flush()
    db.add(AthletePreferences(athlete_id=athlete.id))
    db.add(AthleteStats(athlete_id=athlete.id, threshold_pace=spec.get("threshold_pace", rng.uniform(3.2, 4.5))))
    return athlete


def random_athlete_spec(rng: random.Random, index: int) -> dict:
    first = rng.choice(FIRST_NAMES)
    last = rng.choice(LAST_NAMES)
    city, state = rng.choice(CITIES)
    base = f"{first.lower()}{last.lower()}"
    username = f"{base}{index}"[:64]
    return {
        "username": username,
        "email": f"{username}@runify.test",
        "first_name": first,
        "last_name": last,
        "city": city,
        "state": state,
        "gender": rng.choice(["male", "female", "other"]),
        "birthdate": date(rng.randint(1975, 2002), rng.randint(1, 12), rng.randint(1, 28)),
        "weight_kg": round(rng.uniform(48, 95), 1),
        "height_cm": round(rng.uniform(155, 195), 0),
        "threshold_pace": round(rng.uniform(3.2, 5.0), 2),
    }


def create_rich_activity(
    db,
    rng: random.Random,
    athlete: Athlete,
    *,
    gear_id: str | None = None,
    name: str | None = None,
    start_date: datetime | None = None,
    rich: bool = True,
) -> Activity:
    distance = rng.uniform(3000, 42000)
    pace = rng.uniform(3.2, 6.0)
    elapsed = int((distance / 1000.0) * pace * 60)
    elapsed = max(elapsed, 600)
    start = start_date or random_start_date(rng)
    poly, poly_summary = random_route_polyline(rng)
    activity = Activity(
        athlete_id=athlete.id,
        name=name or rng.choice(ACTIVITY_NAMES),
        description=rng.choice(COMMENT_SNIPPETS) if rng.random() < 0.4 else "",
        activity_type=rng.choice(RUN_TYPES),
        distance=round(distance, 1),
        moving_time=elapsed,
        elapsed_time=elapsed + rng.randint(0, 300),
        start_date=start,
        polyline=poly,
        polyline_summary=poly_summary,
        device_name=rng.choice(DEVICES),
        gear_id=gear_id,
        perceived_exertion=rng.randint(3, 9) if rng.random() < 0.6 else None,
        visibility=rng.choice(["public", "followers", "followers", "private"]),
        biometrics_visibility=rng.choice(["followers", "public", "private"]),
        like_count=0,
        comment_count=0,
    )
    db.add(activity)
    db.flush()

    for metric in build_activity_metrics(rng, activity.distance, pace):
        metric.activity_id = activity.id
        db.add(metric)

    if rich and rng.random() < 0.7:
        for zone in build_zones(rng, elapsed):
            zone.activity_id = activity.id
            db.add(zone)
        for dist in build_distributions(rng, pace, elapsed):
            dist.activity_id = activity.id
            db.add(dist)
        for stream in build_streams(rng, elapsed):
            stream.activity_id = activity.id
            db.add(stream)
        for split in build_splits(activity.distance, elapsed, pace):
            split.activity_id = activity.id
            db.add(split)
        for lap in build_laps(rng, start, activity.distance, elapsed):
            lap.activity_id = activity.id
            db.add(lap)

    stats = db.query(AthleteStats).filter_by(athlete_id=athlete.id).first()
    if stats:
        stats.all_time_run_totals += activity.distance
        stats.ytd_run_totals += activity.distance

    return activity


def seed_follow_graph(db, rng: random.Random, athletes: list[Athlete], demo: Athlete) -> int:
    count = 0
    others = [a for a in athletes if a.id != demo.id]

    # Demo follows everyone featured + random subset
    for target in others[: min(20, len(others))]:
        db.add(Follow(follower_id=demo.id, following_id=target.id, status="following"))
        count += 1

    # Random mutual-ish graph
    for _ in range(len(athletes) * 2):
        follower = rng.choice(athletes)
        following = rng.choice(athletes)
        if follower.id == following.id:
            continue
        status = "pending" if rng.random() < 0.05 else "following"
        db.add(Follow(follower_id=follower.id, following_id=following.id, status=status))
        count += 1

    # Some athletes follow demo back
    for source in rng.sample(others, k=min(15, len(others))):
        db.add(Follow(follower_id=source.id, following_id=demo.id, status="following"))
        count += 1

    db.flush()
    return count


def _seed_comment_likes(
    db,
    rng: random.Random,
    comment: Comment,
    athletes: list[Athlete],
    max_likers: int,
) -> None:
    likers = rng.sample(athletes, k=rng.randint(0, min(max_likers, len(athletes))))
    for liker in likers:
        db.add(Like(athlete_id=liker.id, target_type="comment", target_id=comment.id))
        comment.like_count += 1


def seed_social_engagement(db, rng: random.Random, athletes: list[Athlete], activities: list[Activity]) -> None:
    posts: list[AthletePost] = []
    for athlete in rng.sample(athletes, k=min(30, len(athletes))):
        for _ in range(rng.randint(1, 4)):
            post = AthletePost(
                athlete_id=athlete.id,
                text=rng.choice(POST_SNIPPETS),
                media_urls=[],
                like_count=0,
                comment_count=0,
                created_at=random_start_date(rng, 90),
            )
            db.add(post)
            posts.append(post)
    db.flush()

    public_activities = [a for a in activities if a.visibility in ("public", "followers")]
    sample_acts = rng.sample(public_activities, k=min(200, len(public_activities)))

    for activity in sample_acts:
        likers = rng.sample(athletes, k=rng.randint(0, min(8, len(athletes))))
        for liker in likers:
            db.add(Like(athlete_id=liker.id, target_type="activity", target_id=activity.id))
            activity.like_count += 1

        commenters = rng.sample(athletes, k=rng.randint(0, min(3, len(athletes))))
        for author in commenters:
            comment = Comment(
                author_id=author.id,
                target_type="activity",
                target_id=activity.id,
                text=rng.choice(COMMENT_SNIPPETS),
            )
            db.add(comment)
            activity.comment_count += 1
            db.flush()
            _seed_comment_likes(db, rng, comment, athletes, max_likers=5)

    for post in posts:
        likers = rng.sample(athletes, k=rng.randint(0, min(6, len(athletes))))
        for liker in likers:
            db.add(Like(athlete_id=liker.id, target_type="post", target_id=post.id))
            post.like_count += 1

        commenters = rng.sample(athletes, k=rng.randint(0, min(3, len(athletes))))
        for author in commenters:
            comment = Comment(
                author_id=author.id,
                target_type="post",
                target_id=post.id,
                text=rng.choice(COMMENT_SNIPPETS),
            )
            db.add(comment)
            post.comment_count += 1
            db.flush()
            _seed_comment_likes(db, rng, comment, athletes, max_likers=3)

    club_posts = db.scalars(select(ClubPost)).all()
    for club_post in club_posts:
        likers = rng.sample(athletes, k=rng.randint(0, min(6, len(athletes))))
        for liker in likers:
            db.add(Like(athlete_id=liker.id, target_type="club_post", target_id=club_post.id))
            club_post.like_count += 1

        commenters = rng.sample(athletes, k=rng.randint(0, min(4, len(athletes))))
        for author in commenters:
            comment = Comment(
                author_id=author.id,
                target_type="club_post",
                target_id=club_post.id,
                text=rng.choice(COMMENT_SNIPPETS),
            )
            db.add(comment)
            club_post.comment_count += 1
            db.flush()
            _seed_comment_likes(db, rng, comment, athletes, max_likers=3)

    db.flush()


def seed_clubs(db, rng: random.Random, athletes: list[Athlete], demo: Athlete) -> list[Club]:
    clubs: list[Club] = []
    creators = rng.sample(athletes, k=min(len(CLUB_TEMPLATES), len(athletes)))
    for (name, desc, tags), creator in zip(CLUB_TEMPLATES, creators, strict=False):
        club = Club(
            name=name,
            description=desc,
            tags=tags,
            is_private=rng.random() < 0.15,
            creator_id=creator.id,
            member_count=1,
        )
        db.add(club)
        db.flush()
        db.add(ClubMember(club_id=club.id, athlete_id=creator.id, role="admin"))
        members = rng.sample(athletes, k=rng.randint(5, min(25, len(athletes))))
        for member in members:
            if member.id == creator.id:
                continue
            db.add(ClubMember(club_id=club.id, athlete_id=member.id, role="member"))
            club.member_count += 1
        for _ in range(rng.randint(2, 6)):
            author = rng.choice(members) if members else creator
            db.add(
                ClubPost(
                    club_id=club.id,
                    author_id=author.id,
                    title=rng.choice(["Weekly run", "Race recap", "Route idea", "Meetup"]),
                    body=rng.choice(POST_SNIPPETS),
                    created_at=random_start_date(rng, 60),
                )
            )
        clubs.append(club)

    # Ensure demo is in first club
    if clubs:
        existing = db.query(ClubMember).filter_by(club_id=clubs[0].id, athlete_id=demo.id).first()
        if not existing:
            db.add(ClubMember(club_id=clubs[0].id, athlete_id=demo.id, role="member"))
            clubs[0].member_count += 1

    db.flush()
    return clubs


def seed_segments_and_routes(
    db, rng: random.Random, athletes: list[Athlete], activities: list[Activity], demo: Athlete
) -> tuple[list[Segment], list[Route]]:
    segments: list[Segment] = []
    for name, slat, slng, elat, elng, dist, grade in SEGMENT_TEMPLATES:
        creator = rng.choice(athletes)
        coords = [(slat, slng), ((slat + elat) / 2, (slng + elng) / 2), (elat, elng)]
        seg = Segment(
            name=name,
            distance=dist,
            average_grade=grade,
            start_lat=slat,
            start_lng=slng,
            end_lat=elat,
            end_lng=elng,
            polyline=encode_polyline(coords),
            elevation_high=rng.uniform(50, 250),
            elevation_low=rng.uniform(0, 40),
            creator_id=creator.id,
            total_effort_count=0,
            total_athlete_count=0,
            star_count=0,
        )
        db.add(seg)
        db.flush()
        segments.append(seg)

        starrers = rng.sample(athletes, k=rng.randint(3, min(15, len(athletes))))
        for a in starrers:
            db.add(SegmentStar(segment_id=seg.id, athlete_id=a.id))
            seg.star_count += 1

        effort_acts = rng.sample(activities, k=min(30, len(activities)))
        athletes_seen: set[str] = set()
        for act in effort_acts:
            effort_time = rng.randint(120, 600)
            db.add(
                SegmentEffort(
                    segment_id=seg.id,
                    activity_id=act.id,
                    athlete_id=act.athlete_id,
                    elapsed_time=effort_time,
                    moving_time=effort_time,
                    start_date=act.start_date,
                    average_heartrate=rng.uniform(140, 175),
                )
            )
            seg.total_effort_count += 1
            athletes_seen.add(act.athlete_id)
        seg.total_athlete_count = len(athletes_seen)

    routes: list[Route] = []
    route_owners = rng.sample(athletes, k=min(20, len(athletes)))
    for owner in route_owners:
        for _ in range(rng.randint(1, 3)):
            poly, summary = random_route_polyline(rng, points=rng.randint(25, 60))
            dist = rng.uniform(5000, 25000)
            routes.append(
                Route(
                    athlete_id=owner.id,
                    name=rng.choice(["Scenic loop", "Hill grinder", "Flat fast", "Park perimeter", "Waterfront"]),
                    description=rng.choice(POST_SNIPPETS),
                    distance=round(dist, 1),
                    elevation_gain=rng.uniform(20, 400),
                    polyline=poly,
                    polyline_summary=summary,
                    is_private=rng.random() < 0.2,
                    estimated_duration=int((dist / 1000) * rng.uniform(4, 6) * 60),
                )
            )
    db.add_all(routes)

    # Demo gets a featured route
    demo_poly, demo_summary = random_route_polyline(rng, points=30)
    db.add(
        Route(
            athlete_id=demo.id,
            name="Golden Gate Park loop",
            description="Favorite weekday loop.",
            distance=12450,
            elevation_gain=85,
            polyline=demo_poly,
            polyline_summary=demo_summary,
            is_private=False,
            estimated_duration=3180,
        )
    )

    db.flush()
    return segments, routes


def seed_gear_and_prs(db, rng: random.Random, athletes: list[Athlete], activities: list[Activity]) -> int:
    gear_count = 0
    pr_distances = [
        ("5K", 5000), ("10K", 10000), ("Half Marathon", 21097.5), ("Marathon", 42195),
    ]
    for athlete in athletes:
        for i in range(rng.randint(1, 3)):
            brand, model = rng.choice(SHOE_BRANDS)
            db.add(
                Gear(
                    athlete_id=athlete.id,
                    name=f"{brand} {model}",
                    brand_name=brand,
                    model_name=model,
                    gear_type="shoe",
                    is_primary=i == 0,
                    max_mileage=rng.uniform(400000, 800000),
                    is_retired=rng.random() < 0.1,
                )
            )
            gear_count += 1

        if activities:
            act = rng.choice([a for a in activities if a.athlete_id == athlete.id] or activities)
            dist_name, dist_m = rng.choice(pr_distances)
            pace = rng.uniform(3.0, 5.5)
            time_sec = int((dist_m / 1000) * pace * 60)
            db.add(
                PersonalRecord(
                    athlete_id=athlete.id,
                    distance_name=dist_name,
                    distance_meters=dist_m,
                    time_in_seconds=time_sec,
                    activity_id=act.id,
                    achieved_date=act.start_date,
                )
            )

    db.flush()
    return gear_count


def seed_notifications(db, rng: random.Random, athletes: list[Athlete], demo: Athlete) -> int:
    count = 0
    notif_types = ["like", "comment", "follow", "follow_request", "club_invite"]
    senders = [a for a in athletes if a.id != demo.id]
    for _ in range(40):
        sender = rng.choice(senders)
        ntype = rng.choice(notif_types)
        db.add(
            Notification(
                athlete_id=demo.id,
                type=ntype,
                is_read=rng.random() < 0.35,
                sender_id=sender.id,
                payload={"message": rng.choice(COMMENT_SNIPPETS), "sender_id": sender.id},
                created_at=random_start_date(rng, 30),
            )
        )
        count += 1

    for athlete in rng.sample(athletes, k=min(20, len(athletes))):
        for _ in range(rng.randint(1, 5)):
            sender = rng.choice(athletes)
            if sender.id == athlete.id:
                continue
            db.add(
                Notification(
                    athlete_id=athlete.id,
                    type=rng.choice(notif_types),
                    is_read=rng.random() < 0.5,
                    sender_id=sender.id,
                    payload={"sender_id": sender.id},
                    created_at=random_start_date(rng, 45),
                )
            )
            count += 1

    db.flush()
    return count


def seed_demo_showcase(db, rng: random.Random, demo: Athlete) -> list[Activity]:
    """A few high-quality showcase activities for the demo user."""
    showcase = [
        ("Morning tempo — Golden Gate Park", 12450, 3180, 4.25, 152),
        ("Recovery jog — Embarcadero", 6500, 2340, 6.0, 128),
        ("Hill repeats — Twin Peaks", 8200, 2940, 5.8, 165),
        ("Long run — Marin Headlands", 32100, 10800, 5.6, 142),
    ]
    created: list[Activity] = []
    for i, (name, dist, elapsed, pace, hr) in enumerate(showcase):
        start = utc_now() - timedelta(days=i + 1, hours=6)
        poly, summary = random_route_polyline(rng, points=50)
        activity = Activity(
            athlete_id=demo.id,
            name=name,
            description="Showcase activity for demo account.",
            activity_type="run",
            distance=float(dist),
            moving_time=elapsed,
            elapsed_time=elapsed + 120,
            start_date=start,
            polyline=poly,
            polyline_summary=summary,
            device_name="Garmin Forerunner 265",
            perceived_exertion=6 + i % 3,
            visibility="followers",
            biometrics_visibility="followers",
        )
        db.add(activity)
        db.flush()
        for metric in build_activity_metrics(rng, dist, pace):
            metric.activity_id = activity.id
            if metric.key == "avg_hr":
                metric.value = hr
            db.add(metric)
        for zone in build_zones(rng, elapsed):
            zone.activity_id = activity.id
            db.add(zone)
        for dist_obj in build_distributions(rng, pace, elapsed):
            dist_obj.activity_id = activity.id
            db.add(dist_obj)
        for stream in build_streams(rng, elapsed):
            stream.activity_id = activity.id
            db.add(stream)
        for split in build_splits(dist, elapsed, pace):
            split.activity_id = activity.id
            db.add(split)
        stats = db.query(AthleteStats).filter_by(athlete_id=demo.id).first()
        if stats:
            stats.all_time_run_totals += dist
            stats.ytd_run_totals += dist
        created.append(activity)
    db.flush()
    return created


def run_seed(
    *,
    athletes_count: int = 50,
    activities_per_athlete: int = 15,
    clear: bool = False,
    seed: int = 7,
) -> None:
    rng = random.Random(seed)
    db = SessionLocal()
    try:
        ensure_schema(db)
        if clear:
            clear_database(db)
        elif db.query(Athlete).count() > 5:
            print(
                "Database already has data. Pass --clear to wipe and reseed, "
                "or use fewer existing athletes."
            )
            return

        print(f"Seeding with seed={seed}, athletes={athletes_count}, activities/athlete={activities_per_athlete}")

        athletes: list[Athlete] = []
        for spec in FEATURED_ATHLETES:
            athletes.append(create_athlete(db, rng, spec))

        extra = max(0, athletes_count - len(FEATURED_ATHLETES))
        for i in range(extra):
            athletes.append(create_athlete(db, rng, random_athlete_spec(rng, i)))

        demo = next(a for a in athletes if a.username == "demo")
        db.flush()

        all_activities: list[Activity] = []
        all_activities.extend(seed_demo_showcase(db, rng, demo))

        for athlete in athletes:
            n = activities_per_athlete if athlete.id != demo.id else max(5, activities_per_athlete // 2)
            for _ in range(n):
                all_activities.append(create_rich_activity(db, rng, athlete, rich=rng.random() < 0.75))

        gear_count = seed_gear_and_prs(db, rng, athletes, all_activities)
        follow_count = seed_follow_graph(db, rng, athletes, demo)
        clubs = seed_clubs(db, rng, athletes, demo)
        segments, routes = seed_segments_and_routes(db, rng, athletes, all_activities, demo)
        seed_social_engagement(db, rng, athletes, all_activities)
        notif_count = seed_notifications(db, rng, athletes, demo)

        db.commit()

        print()
        print("Seed complete!")
        print(f"  Athletes:    {len(athletes)}")
        print(f"  Activities:  {len(all_activities)}")
        print(f"  Gear items:  {gear_count}")
        print(f"  Follows:     {follow_count}+")
        print(f"  Clubs:       {len(clubs)}")
        print(f"  Segments:    {len(segments)}")
        print(f"  Routes:      {len(routes)}+")
        print(f"  Notifications: {notif_count}")
        print()
        print("Login: username=demo  password=demo1234")
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Seed Runify with bulk random test data")
    parser.add_argument("--athletes", type=int, default=50, help="Total athletes to create (default: 50)")
    parser.add_argument(
        "--activities-per-athlete",
        type=int,
        default=15,
        help="Activities per athlete (default: 15)",
    )
    parser.add_argument("--clear", action="store_true", help="Truncate all tables before seeding")
    parser.add_argument("--seed", type=int, default=7, help="Random seed for reproducibility (default: 7)")
    args = parser.parse_args()
    run_seed(
        athletes_count=args.athletes,
        activities_per_athlete=args.activities_per_athlete,
        clear=args.clear,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
