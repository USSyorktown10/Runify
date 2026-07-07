from datetime import datetime, timezone
from pathlib import Path

import polyline as polyline_lib
from sqlalchemy.orm import Session

from app.models.activity import (
    Activity,
    ActivityDistribution,
    ActivityLap,
    ActivityMetric,
    ActivitySplit,
    ActivityStream,
    ActivityZone,
    Upload,
)
from app.models.athlete import AthleteStats, PersonalRecord
from app.services.metrics_engine.formulas import (
    ParsedActivity,
    build_distributions,
    build_hr_zones,
    build_pace_zones,
    downsample,
    estimate_vo2max,
    grade_adjusted_speed,
    grade_percent,
    normalized_power,
)
from app.services.metrics_engine.parsers.fit_parser import parse_fit
from app.services.metrics_engine.parsers.gpx_parser import parse_gpx
from app.services.metrics_engine.parsers.tcx_parser import parse_tcx

STANDARD_DISTANCES = [
    ("1K", 1000),
    ("5K", 5000),
    ("10K", 10000),
    ("Half Marathon", 21097.5),
    ("Marathon", 42195),
]


def parse_file(file_path: str) -> ParsedActivity:
    ext = Path(file_path).suffix.lower()
    if ext == ".fit":
        return parse_fit(file_path)
    if ext == ".gpx":
        return parse_gpx(file_path)
    if ext == ".tcx":
        return parse_tcx(file_path)
    raise ValueError(f"Unsupported file type: {ext}")


def encode_polyline(points: list) -> str:
    coords = [(p.lat, p.lng) for p in points if p.lat is not None and p.lng is not None]
    if not coords:
        return ""
    return polyline_lib.encode(coords)


def process_upload(db: Session, upload: Upload, threshold_pace: float = 3.5, max_hr: float = 190.0) -> Activity:
    parsed = parse_file(upload.file_path)
    points = parsed.points
    if not points:
        raise ValueError("No track points found in file")

    elapsed = int(points[-1].time)
    distance = points[-1].distance or 0.0
    moving_time = elapsed

    activity = Activity(
        athlete_id=upload.athlete_id,
        name=parsed.name,
        activity_type=parsed.activity_type,
        distance=distance,
        moving_time=moving_time,
        elapsed_time=elapsed,
        start_date=datetime.fromisoformat(parsed.start_date.replace("Z", "+00:00")),
        polyline=encode_polyline(points),
        polyline_summary=encode_polyline(points[:: max(1, len(points) // 50)]),
        device_name=parsed.device_name,
        raw_file_path=upload.file_path,
    )
    db.add(activity)
    db.flush()

    times = []
    for i, _ in enumerate(points):
        if i == 0:
            times.append(1.0)
        else:
            times.append(max(points[i].time - points[i - 1].time, 0.5))

    stream_defs = {
        "lat": [p.lat for p in points if p.lat is not None],
        "lng": [p.lng for p in points if p.lng is not None],
        "altitude": [p.altitude for p in points if p.altitude is not None],
        "heart_rate": [p.heart_rate for p in points if p.heart_rate is not None],
        "cadence": [p.cadence for p in points if p.cadence is not None],
        "power": [p.power for p in points if p.power is not None],
        "distance": [p.distance for p in points if p.distance is not None],
    }
    axis_time = [p.time for p in points]
    axis_dist = [p.distance or 0 for p in points]

    gap_values = []
    for i in range(1, len(points)):
        dist = (points[i].distance or 0) - (points[i - 1].distance or 0)
        dt = points[i].time - points[i - 1].time
        speed = dist / dt if dt > 0 else 0
        g = grade_percent(points[i - 1].altitude, points[i].altitude, dist)
        gap_values.append(grade_adjusted_speed(speed, g))
    if gap_values:
        stream_defs["grade_adjusted_pace"] = gap_values

    for key, data in stream_defs.items():
        if not data:
            continue
        axis = axis_dist if key in ("distance", "grade_adjusted_pace") else axis_time[: len(data)]
        for res_name, target in [("high", len(data)), ("medium", 200), ("low", 50)]:
            d, a = downsample(data, axis, min(target, len(data)))
            db.add(
                ActivityStream(
                    activity_id=activity.id,
                    metric_key=key,
                    stream_type="calculated" if key == "grade_adjusted_pace" else "raw",
                    axis_type="distance" if key in ("distance", "grade_adjusted_pace") else "time",
                    resolution=res_name,
                    original_size=len(data),
                    data=d,
                    axis=a,
                )
            )

    hr_vals = [p.heart_rate for p in points if p.heart_rate]
    power_vals = [p.power for p in points if p.power]
    speeds = []
    for i in range(1, len(points)):
        dist = (points[i].distance or 0) - (points[i - 1].distance or 0)
        dt = points[i].time - points[i - 1].time
        speeds.append(dist / dt if dt > 0 else 0)
    avg_speed = distance / moving_time if moving_time > 0 else 0

    metrics = [
        ("distance", distance, "raw", "m", "Distance"),
        ("moving_time", moving_time, "raw", "s", "Moving Time"),
        ("avg_speed", avg_speed, "calculated", "m/s", "Average Speed"),
    ]
    if hr_vals:
        metrics.extend([
            ("avg_hr", sum(hr_vals) / len(hr_vals), "calculated", "bpm", "Average Heart Rate"),
            ("max_hr", max(hr_vals), "raw", "bpm", "Max Heart Rate"),
        ])
    if power_vals:
        np_val = normalized_power(power_vals)
        metrics.extend([
            ("normalized_power", np_val, "calculated", "watts", "Normalized Power"),
            ("avg_power", sum(power_vals) / len(power_vals), "calculated", "watts", "Average Power"),
        ])
    if speeds and hr_vals:
        vo2 = estimate_vo2max(avg_speed, sum(hr_vals) / len(hr_vals), max_hr)
        metrics.append(("vo2_max", vo2, "calculated", "ml/kg/min", "VO2 Max"))

  # elevation gain
    elev_gain = 0.0
    for i in range(1, len(points)):
        if points[i].altitude is not None and points[i - 1].altitude is not None:
            diff = points[i].altitude - points[i - 1].altitude
            if diff > 0:
                elev_gain += diff
    metrics.append(("elevation_gain", elev_gain, "calculated", "m", "Elevation Gain"))

    for key, value, source, unit, display_name in metrics:
        db.add(
            ActivityMetric(
                activity_id=activity.id,
                key=key,
                value=value,
                source=source,
                unit=unit,
                display_name=display_name,
            )
        )

    if hr_vals:
        hr_times = times[: len(hr_vals)]
        buckets = [(100 + i * 10, 110 + i * 10) for i in range(10)]
        db.add(
            ActivityDistribution(
                activity_id=activity.id,
                key="heart_rate",
                display_name="Heart Rate Distribution",
                unit="bpm",
                buckets=build_distributions(hr_vals, hr_times, buckets),
            )
        )
        db.add(
            ActivityZone(
                activity_id=activity.id,
                key="heart_rate",
                display_name="Heart Rate Zones",
                unit="bpm",
                reference_value=max_hr,
                reference_name="Max Heart Rate",
                zones=build_hr_zones(hr_vals, hr_times, max_hr),
            )
        )

    pace_vals = [1000 / (s * 60) if s > 0 else 0 for s in speeds]
    if pace_vals:
        db.add(
            ActivityZone(
                activity_id=activity.id,
                key="pace",
                display_name="Pace Zones",
                unit="min/km",
                reference_value=threshold_pace,
                reference_name="Threshold Pace",
                zones=build_pace_zones(pace_vals, times[1 : len(pace_vals) + 1], threshold_pace),
            )
        )

    if power_vals:
        db.add(
            ActivityZone(
                activity_id=activity.id,
                key="power",
                display_name="Power Zones",
                unit="watts",
                reference_value=250.0,
                reference_name="FTP",
                zones=build_hr_zones(power_vals, times[: len(power_vals)], 300),
            )
        )

    for i, lap in enumerate(parsed.laps):
        db.add(
            ActivityLap(
                activity_id=activity.id,
                lap_index=i,
                name=f"Lap {i + 1}",
                start_date=activity.start_date,
                elapsed_time=lap["elapsed_time"],
                moving_time=lap["moving_time"],
                distance=lap["distance"],
                average_speed=lap["average_speed"],
            )
        )

    split_dist = 1000.0
    split_idx = 0
    last_dist = 0.0
    last_time = 0.0
    last_alt = points[0].altitude or 0.0
    for p in points:
        d = p.distance or 0
        if d - last_dist >= split_dist:
            split_idx += 1
            dt = int(p.time - last_time)
            elev_diff = (p.altitude or 0) - last_alt
            avg_spd = split_dist / dt if dt > 0 else 0
            db.add(
                ActivitySplit(
                    activity_id=activity.id,
                    index=split_idx,
                    distance=split_dist,
                    elapsed_time=dt,
                    elevation_difference=elev_diff,
                    average_speed=avg_spd,
                )
            )
            last_dist = d
            last_time = p.time
            last_alt = p.altitude or last_alt

    _update_athlete_totals(db, upload.athlete_id, distance)
    _update_personal_records(db, upload.athlete_id, activity.id, distance, moving_time, activity.start_date)

    upload.status = "completed"
    upload.activity_id = activity.id
    upload.completed_at = datetime.now(timezone.utc)
    db.flush()
    return activity


def _update_athlete_totals(db: Session, athlete_id: str, distance: float) -> None:
    stats = db.query(AthleteStats).filter_by(athlete_id=athlete_id).first()
    if stats:
        stats.all_time_run_totals += distance
        stats.ytd_run_totals += distance


def _update_personal_records(
    db: Session, athlete_id: str, activity_id: str, distance: float, time_s: int, achieved: datetime
) -> None:
    for name, target_dist in STANDARD_DISTANCES:
        if abs(distance - target_dist) / target_dist < 0.02:
            existing = (
                db.query(PersonalRecord)
                .filter_by(athlete_id=athlete_id, distance_name=name)
                .first()
            )
            if not existing or time_s < existing.time_in_seconds:
                if existing:
                    existing.time_in_seconds = time_s
                    existing.activity_id = activity_id
                    existing.achieved_date = achieved
                else:
                    db.add(
                        PersonalRecord(
                            athlete_id=athlete_id,
                            distance_name=name,
                            distance_meters=target_dist,
                            time_in_seconds=time_s,
                            activity_id=activity_id,
                            achieved_date=achieved,
                        )
                    )
