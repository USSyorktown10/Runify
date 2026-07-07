from formulas import grade_percent, minetti_cost, grade_adjusted_speed, haversine, compute_distances
import gpxpy
import statistics
import matplotlib.pyplot as plt

# input 
def gpx_to_gap(gpx_file: str) -> list:
    with open(gpx_file, 'r') as f:
        gpx = gpxpy.parse(f)

    elev = []
    latlng = []
    for track in gpx.tracks:
        for segment in track.segments:
            for point in segment.points:
                elev.append(point.elevation)
                latlng.append([point.latitude, point.longitude])

    haversine_results = []     
    for i in range(len(latlng) - 1):
        haversine_results.append(haversine(latlng[i][0], latlng[i][1], latlng[i+1][0], latlng[i+1][1]))
    print(sum(haversine_results))
    speeds = []
    grades = []
    gap_speeds = []
    for i in range(len(latlng) - 1):
        grades.append(grade_percent(elev[i], elev[i+1], haversine_results[i]))
        p1 = gpx.tracks[0].segments[0].points[i]
        p2 = gpx.tracks[0].segments[0].points[i+1]
        if p1.time and p2.time and haversine_results[i] > 0:
            time_delta = (p2.time - p1.time).total_seconds()
            if time_delta > 0:
                actual_speed = haversine_results[i] / time_delta
            else:
                actual_speed = 0.0
        else:
            actual_speed = 0.0
        speeds.append(actual_speed)
        gap_speeds.append(grade_adjusted_speed(speeds[i], grades[i]))
    return gap_speeds
