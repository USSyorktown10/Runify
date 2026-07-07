from datetime import datetime, timezone

import fitparse

from app.services.metrics_engine.formulas import (
    ParsedActivity,
    ParsedPoint,
    compute_distances,
)


def parse_fit(file_path: str) -> ParsedActivity:
    fit = fitparse.FitFile(str(file_path))
    points: list[ParsedPoint] = []
    start_date = ""
    device_name = ""
    laps: list[dict] = []

    for record in fit.get_messages("file_id"):
        for field in record:
            if field.name == "time_created" and field.value:
                start_date = field.value.isoformat()
            if field.name == "product_name" and field.value:
                device_name = str(field.value)

    t0 = None
    for record in fit.get_messages("record"):
        data: dict = {}
        for field in record:
            data[field.name] = field.value
        ts = data.get("timestamp")
        if ts is None:
            continue
        if t0 is None:
            t0 = ts
        elapsed = (ts - t0).total_seconds()
        points.append(
            ParsedPoint(
                time=elapsed,
                lat=data.get("position_lat"),
                lng=data.get("position_long"),
                altitude=data.get("altitude"),
                heart_rate=data.get("heart_rate"),
                cadence=data.get("cadence"),
                power=data.get("power"),
                speed=data.get("enhanced_speed") or data.get("speed"),
            )
        )

    for lap_msg in fit.get_messages("lap"):
        lap_data: dict = {}
        for field in lap_msg:
            lap_data[field.name] = field.value
        laps.append(
            {
                "elapsed_time": int(lap_data.get("total_elapsed_time", 0)),
                "moving_time": int(lap_data.get("total_timer_time", 0)),
                "distance": float(lap_data.get("total_distance", 0)),
                "average_speed": float(lap_data.get("enhanced_avg_speed") or lap_data.get("avg_speed") or 0),
            }
        )

    if points and points[0].lat and abs(points[0].lat) > 1000:
        for p in points:
            if p.lat:
                p.lat = p.lat * (180 / 2**31)
            if p.lng:
                p.lng = p.lng * (180 / 2**31)

    compute_distances(points)
    if not start_date:
        start_date = datetime.now(timezone.utc).isoformat()

    return ParsedActivity(
        name="FIT Import",
        start_date=start_date,
        device_name=device_name,
        points=points,
        laps=laps,
    )
