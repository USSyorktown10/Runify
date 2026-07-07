import math

# input lat and long, outputs meters away
def haversine(lat1: float, lng1: float, lat2: float, lng2: float) -> float:
    r = 6371000
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lng2 - lng1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    return 2 * r * math.atan2(math.sqrt(a), math.sqrt(1 - a))

# inputs lat and lon pairs, returns nothing but changes list to include dist in meters between every pt
def compute_distances(points: list) -> None:
    total = 0.0
    for i, p in enumerate(points):
        if i == 0:
            p.distance = 0.0
            continue
        prev = points[i - 1]
        if p.lat is not None and p.lng is not None and prev.lat is not None and prev.lng is not None:
            total += haversine(prev.lat, prev.lng, p.lat, p.lng)
        p.distance = total

# input unit-agnostic, returns percentage
def grade_percent(alt1: float | None, alt2: float | None, dist: float) -> float:
    if alt1 is None or alt2 is None or dist <= 0:
        return 0.0
    return ((alt2 - alt1) / dist) * 100

# % input (10% would make an input of 10.0), returns watts/kg
def minetti_cost(grade_pct: float) -> float:
    g = grade_pct / 100
    return (155.4 * (g**5) - 30.4 * (g**4) - 43.3 * (g**3) + 46.3 * (g**2) + 19.5 * g + 3.6) / 3.6

# input unit-agnistic speed and grade as %, returns grade adjusted speed using input unit
def grade_adjusted_speed(speed: float, grade_pct: float) -> float:
    if speed <= 0:
        return 0.0
    cost = minetti_cost(grade_pct)
    return speed / max(cost, 0.1)

def normalized_power(power_values: list[float], sample_rate: float = 1.0) -> float:
    if not power_values:
        return 0.0
    window = max(1, int(30 * sample_rate))
    rolling: list[float] = []
    for i in range(len(power_values)):
        start = max(0, i - window + 1)
        chunk = power_values[start : i + 1]
        rolling.append(sum(chunk) / len(chunk))
    fourth = [r**4 for r in rolling]
    return (sum(fourth) / len(fourth)) ** 0.25


def estimate_vo2max(speed_mps: float, hr: float, max_hr: float = 190.0) -> float:
    # Daniels-Gilbert simplified estimate
    if speed_mps <= 0 or hr <= 0:
        return 0.0
    pct_hr = min(hr / max_hr, 0.95)
    vo2 = -4.6 + 0.182258 * (speed_mps * 60) + 0.000104 * (speed_mps * 60) ** 2
    return vo2 / max(pct_hr, 0.5)


def build_distributions(values: list[float], times: list[float], buckets: list[tuple[float, float]]) -> list[dict]:
    result = []
    for lo, hi in buckets:
        t = 0.0
        for v, dt in zip(values, times, strict=False):
            if lo <= v < hi:
                t += dt
        result.append({"min_value": lo, "max_value": hi, "time_in_seconds": int(t)})
    return result


def build_hr_zones(hr_values: list[float], times: list[float], max_hr: float) -> list[dict]:
    zones_def = [(0.5, 0.6), (0.6, 0.7), (0.7, 0.8), (0.8, 0.9), (0.9, 1.01)]
    zones = []
    for i, (lo_pct, hi_pct) in enumerate(zones_def, 1):
        lo, hi = max_hr * lo_pct, max_hr * hi_pct
        t = sum(dt for v, dt in zip(hr_values, times, strict=False) if lo <= v < hi)
        zones.append({"zone_index": i, "min_value": lo, "max_value": hi, "time_in_seconds": int(t)})
    return zones


def build_pace_zones(pace_values: list[float], times: list[float], threshold_pace: float) -> list[dict]:
    # pace in min/km; threshold_pace is m/s
    threshold_min_per_km = 1000 / (threshold_pace * 60) if threshold_pace > 0 else 6.0
    factors = [1.3, 1.15, 1.0, 0.92, 0.85]
    zones = []
    for i, f in enumerate(factors, 1):
        center = threshold_min_per_km * f
        lo = center - 0.25
        hi = center + 0.25
        t = sum(dt for v, dt in zip(pace_values, times, strict=False) if lo <= v < hi)
        zones.append({"zone_index": i, "min_value": lo, "max_value": hi, "time_in_seconds": int(t)})
    return zones


def power_curve(power_values: list[float], sample_rate: float = 1.0) -> list[dict]:
    durations = [1, 5, 10, 30, 60, 120, 300, 600, 1200, 3600]
    result = []
    for d in durations:
        window = max(1, int(d * sample_rate))
        best = 0.0
        for i in range(len(power_values) - window + 1):
            avg = sum(power_values[i : i + window]) / window
            best = max(best, avg)
        result.append({"time_interval_seconds": d, "power_value_watts": best})
    return result


def downsample(data: list[float], axis: list[float], target: int) -> tuple[list[float], list[float]]:
    if len(data) <= target:
        return data, axis
    step = len(data) / target
    new_data, new_axis = [], []
    for i in range(target):
        idx = int(i * step)
        new_data.append(data[idx])
        new_axis.append(axis[idx])
    return new_data, new_axis
