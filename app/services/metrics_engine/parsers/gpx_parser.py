from datetime import datetime, timezone

import gpxpy

from app.services.metrics_engine.formulas import (
    ParsedActivity,
    ParsedPoint,
    compute_distances,
)


def parse_gpx(file_path: str) -> ParsedActivity:
    with open(file_path) as f:
        gpx = gpxpy.parse(f)

    points: list[ParsedPoint] = []
    start_date = ""
    t0 = None

    for track in gpx.tracks:
        for segment in track.segments:
            for pt in segment.points:
                if not start_date and pt.time:
                    start_date = pt.time.isoformat()
                elapsed = 0.0
                if pt.time:
                    if t0 is None:
                        t0 = pt.time
                    elapsed = (pt.time - t0).total_seconds()
                points.append(
                    ParsedPoint(
                        time=elapsed,
                        lat=pt.latitude,
                        lng=pt.longitude,
                        altitude=pt.elevation,
                    )
                )

    compute_distances(points)
    if not start_date:
        start_date = datetime.now(timezone.utc).isoformat()

    return ParsedActivity(name="GPX Import", start_date=start_date, points=points)
