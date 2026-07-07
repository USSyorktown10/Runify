import xml.etree.ElementTree as ET
from datetime import datetime, timezone

from app.services.metrics_engine.formulas import (
    ParsedActivity,
    ParsedPoint,
    compute_distances,
)

TCX_NS = {"tcx": "http://www.garmin.com/xmlschemas/TrainingCenterDatabase/v2"}


def parse_tcx(file_path: str) -> ParsedActivity:
    tree = ET.parse(file_path)
    root = tree.getroot()
    points: list[ParsedPoint] = []
    start_date = ""
    t0 = None

    for trackpoint in root.findall(".//tcx:Trackpoint", TCX_NS):
        time_el = trackpoint.find("tcx:Time", TCX_NS)
        pos_el = trackpoint.find("tcx:Position", TCX_NS)
        alt_el = trackpoint.find("tcx:AltitudeMeters", TCX_NS)
        hr_el = trackpoint.find(".//tcx:HeartRateBpm/tcx:Value", TCX_NS)
        cad_el = trackpoint.find("tcx:Cadence", TCX_NS)

        if time_el is None or time_el.text is None:
            continue
        ts = datetime.fromisoformat(time_el.text.replace("Z", "+00:00"))
        if not start_date:
            start_date = ts.isoformat()
        if t0 is None:
            t0 = ts
        elapsed = (ts - t0).total_seconds()

        lat = lng = None
        if pos_el is not None:
            lat_el = pos_el.find("tcx:LatitudeDegrees", TCX_NS)
            lng_el = pos_el.find("tcx:LongitudeDegrees", TCX_NS)
            if lat_el is not None and lat_el.text:
                lat = float(lat_el.text)
            if lng_el is not None and lng_el.text:
                lng = float(lng_el.text)

        alt = float(alt_el.text) if alt_el is not None and alt_el.text else None
        hr = float(hr_el.text) if hr_el is not None and hr_el.text else None
        cad = float(cad_el.text) if cad_el is not None and cad_el.text else None

        points.append(ParsedPoint(time=elapsed, lat=lat, lng=lng, altitude=alt, heart_rate=hr, cadence=cad))

    compute_distances(points)
    if not start_date:
        start_date = datetime.now(timezone.utc).isoformat()

    return ParsedActivity(name="TCX Import", start_date=start_date, points=points)
