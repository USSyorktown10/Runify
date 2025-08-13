import math
import joblib
from datetime import datetime, timezone
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import List

'''
Testing new formulas i have discovered
'''

def minetti_grade_adjustment(grade_percent):
    """
    Minetti formula for metabolic cost adjustment based on grade
    """
    i = grade_percent / 100  # Convert percentage to decimal
    cost_factor = 155.4 * (i**5) - 30.4 * (i**4) - 43.3 * (i**3) + 46.3 * (i**2) - 165 * i + 3.6
    return cost_factor / 100  # Normalize to multiplier

def strava_grade_adjustment(grade_percent):
    """
    Strava gets robbed
    """
    i = grade_percent / 100
    relative_cost = 15.14 * (i**2) - 2.896 * i
    return 1 + (relative_cost / 100)

def simple_grade_adjustment(grade_percent):
    """
    Simple linear approximation: 10% grade = 30% harder
    """
    return 1 + (abs(grade_percent) * 0.03)

def calculate_gap(pace_per_km, elevation_gain_m, distance_km):
    """
    pace_per_km: seconds per kilometer
    elevation_gain_m: total elevation gain in meters
    distance_km: distance in kilometers
    """
    if distance_km == 0:
        return pace_per_km
    
    avg_grade_percent = (elevation_gain_m / (distance_km * 1000)) * 100
    
    # Use simple adjustment for now
    adjustment_factor = simple_grade_adjustment(avg_grade_percent)
    
    # GAP is the equivalent flat pace
    gap = pace_per_km / adjustment_factor
    return gap

def calculate_ngp(pace_data_per_km, elevation_data_m, distance_data_km):
    if not pace_data_per_km or len(pace_data_per_km) == 0:
        return 0
    
    # Calculate GAP for each segment
    gap_values = []
    for i in range(len(pace_data_per_km)):
        pace = pace_data_per_km[i]
        elevation = elevation_data_m[i] if i < len(elevation_data_m) else 0
        distance = distance_data_km[i] if i < len(distance_data_km) else 1
        
        gap = calculate_gap(pace, elevation, distance)
        gap_values.append(gap)
    
    # Convert pace to speed (km/h) for 4th power calculation
    speeds = [3600 / gap for gap in gap_values if gap > 0]
    
    if not speeds:
        return 0
    
    # 4th power weighting (like Normalized Power)
    fourth_powers = [speed**4 for speed in speeds]
    avg_fourth_power = sum(fourth_powers) / len(fourth_powers)
    normalized_speed = avg_fourth_power ** 0.25
    
    # Convert back to pace (seconds per km)
    ngp = 3600 / normalized_speed if normalized_speed > 0 else 0
    return ngp

def calculate_intensity_factor(ngp_pace_per_km, threshold_pace_per_km):
    if threshold_pace_per_km <= 0 or ngp_pace_per_km <= 0:
        return 0
    
    # Convert to speeds for proper ratio
    ngp_speed = 3600 / ngp_pace_per_km
    threshold_speed = 3600 / threshold_pace_per_km
    
    intensity_factor = ngp_speed / threshold_speed
    return intensity_factor

def calculate_rtss(duration_seconds, ngp_pace_per_km, threshold_pace_per_km):
    if threshold_pace_per_km <= 0 or ngp_pace_per_km <= 0:
        return 0
    
    intensity_factor = calculate_intensity_factor(ngp_pace_per_km, threshold_pace_per_km)
    duration_hours = duration_seconds / 3600
    
    rtss = duration_hours * (intensity_factor ** 2) * 100
    
    return rtss

def calculate_rtss_alternative(duration_seconds, ngp_pace_per_km, threshold_pace_per_km):
    if threshold_pace_per_km <= 0 or ngp_pace_per_km <= 0:
        return 0
    
    intensity_factor = calculate_intensity_factor(ngp_pace_per_km, threshold_pace_per_km)
    
    # (duration * ngp * intensity_factor) / (ftp * 3600) * 100
    rtss = (duration_seconds * ngp_pace_per_km * intensity_factor) / (threshold_pace_per_km * 3600) * 100
    
    return rtss

# Example usage and testing
if __name__ == "__main__":
    # Test data
    pace_data = [300, 320, 310]  # seconds per km
    elevation_data = [50, 100, 30]  # meters elevation gain per segment
    distance_data = [1, 1, 1]  # km per segment
    
    total_duration = 930  # seconds (15.5 minutes)
    threshold_pace = 240  # seconds per km (4:00/km threshold)
    
    print("=== Testing rTSS Calculations ===")
    
    # Calculate NGP
    ngp = calculate_ngp(pace_data, elevation_data, distance_data)
    print(f"NGP: {ngp:.1f} seconds/km ({ngp/60:.2f} min/km)")
    
    # Calculate IF
    intensity_factor = calculate_intensity_factor(ngp, threshold_pace)
    print(f"Intensity Factor: {intensity_factor:.3f}")
    
    # Calculate rTSS both ways
    rtss1 = calculate_rtss(total_duration, ngp, threshold_pace)
    rtss2 = calculate_rtss_alternative(total_duration, ngp, threshold_pace)
    
    print(f"rTSS (standard formula): {rtss1:.1f}")
    print(f"rTSS (secondary formula): {rtss2:.1f}")
    
    print(f"\nDuration: {total_duration/60:.1f} minutes")
    print(f"Threshold pace: {threshold_pace/60:.2f} min/km")

'''
The following are my simpler fomulas that im still working on.
'''

CTLmin = 0
CTLmax = 150
ATLmin = 0
ATLmax = 150
Tfit = 42
Tfat = 7
ISA_SEA_LEVEL_TEMP_C = 15.0
ISA_SEA_LEVEL_PRESSURE_PA = 101325.0
LAPSE_RATE = 0.0065
TROPOPAUSE_ALT = 11000.0
TROPOPAUSE_TEMP_C = -56.5
GAS_CONST = 287.05
GRAVITY = 9.80665
ISA_SEA_LEVEL_TEMP_K = 288.15
MOLAR_MASS_AIR = 0.0289644  # kg/mol (Molar mass of dry air)
lapse_rate = 0.0065  # K/m (approximate lapse rate in the troposphere)

# Calculate EXPONENT based on the barometric formula for the troposphere
EXPONENT = (GRAVITY * MOLAR_MASS_AIR) / (GAS_CONST * lapse_rate)

def wtss(duration, normalized_power, intensity_factor, ftp): #duration must be seconds
    tss = (duration * normalized_power * intensity_factor) / (ftp * 3600) * 100
    return tss

def rtss(duration, ngp, intensity_factor, ftp): #duration must be seconds
    tss = (duration * ngp * intensity_factor) / (ftp * 3600) * 100
    return tss