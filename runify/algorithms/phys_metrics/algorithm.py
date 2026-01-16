import math
import joblib
from datetime import datetime, timezone
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import List
import joblib as _joblib
from typing import Dict, Any
from scipy.signal import lfilter

# local imports for model helpers
try:
    # package-style import
    from .vo2_model import load_pipeline as _load_vo2_pipeline, predict_vo2_from_row as _predict_vo2_from_row, predict_vo2_refined as _predict_vo2_refined
except Exception:
    # script-style import
    from vo2_model import load_pipeline as _load_vo2_pipeline, predict_vo2_from_row as _predict_vo2_from_row, predict_vo2_refined as _predict_vo2_refined

# Pre-define stuff

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
EXPONENT = (GRAVITY * MOLAR_MASS_AIR) / (GAS_CONST * lapse_rate) # Calculate EXPONENT based on the barometric formula for the troposphere

'''
The following are new formulas that being tested and refined.
'''

# --- The following are formulas for grade adjustment ---

def parse_time_to_seconds(time_str):
    """Parses 'HH:MM:SS' or 'MM:SS' into total seconds."""
    if pd.isna(time_str) or str(time_str).strip() in ['--', '']:
        return 0.0
    try:
        parts = str(time_str).split(':')
        if len(parts) == 3: # HH:MM:SS
            return int(parts[0])*3600 + int(parts[1])*60 + float(parts[2])
        elif len(parts) == 2: # MM:SS
            return int(parts[0])*60 + float(parts[1])
        return float(time_str)
    except:
        return 0.0

def minetti_grade_adjustment(grade_percent):
    """
    Minetti (2002) Energy Cost of Running.
    Returns J/kg/m (Joules per kg per meter).
    """
    i = grade_percent / 100.0 
    cost = 155.4*(i**5) - 30.4*(i**4) - 43.3*(i**3) + 46.3*(i**2) + 19.5*i + 3.6
    
    # Safety floor: Running downhill never becomes "free" energy generation
    return max(cost, 1.0)

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

# --- End Grade Adjustment Section ---

# --- GAP Calculation ---

def calculate_gap(pace_per_km, grade_percent):
    """
    Refined GAP using Minetti ratio.
    Accepts grade_percent directly (e.g., 5.0 for 5% slope).
    """
    if pace_per_km <= 0: return 0
    
    # Cost at current grade
    cost_at_grade = minetti_grade_adjustment(grade_percent)
    
    # Cost at flat (0%)
    cost_at_flat = minetti_grade_adjustment(0)
    
    # Ratio: How much harder is this than flat?
    adjustment_factor = cost_at_grade / cost_at_flat
    
    # GAP = Actual Pace / Factor (Faster pace for harder work)
    return pace_per_km / adjustment_factor


def calculate_ngp(pace_input, grade_input):
    """
    Calculates Normalized Graded Pace (NGP).
    SMART LOGIC: Detects if input is a Stream (List) or Summary (Float).
    """
    # CASE 1: Summary Data (Float/Int) - Used by your CSV Trainer
    # If the input is just a number (e.g., 300 seconds), treat it as a summary
    if isinstance(pace_input, (int, float, np.number)) and isinstance(grade_input, (int, float, np.number)):
        return calculate_gap(pace_input, grade_input)

    # CASE 2: Stream Data (List/Array) - Used if you parse FIT files later
    if not pace_input: return 0
    
    gap_values = []
    # Use zip for safety if lists are different lengths
    length = min(len(pace_input), len(grade_input))
    
    for i in range(length):
        p = pace_input[i]
        g = grade_input[i]
        gap_values.append(calculate_gap(p, g))
        
    # Weighting Logic (Fourth Power of Speed)
    speeds_mps = [1000/gap for gap in gap_values if gap > 0]
    
    if not speeds_mps: return 0
    
    sum_fourth = sum([s**4 for s in speeds_mps])
    avg_fourth = sum_fourth / len(speeds_mps)
    normalized_speed = avg_fourth ** 0.25
    
    if normalized_speed > 0:
        return 1000 / normalized_speed
    return 0

# --- End GAP Calculation ---

def calculate_intensity_factor(ngp_pace_per_km, threshold_pace_per_km):
    if threshold_pace_per_km <= 0 or ngp_pace_per_km <= 0:
        return 0
    
    # Convert to speeds for proper ratio
    ngp_speed = 3600 / ngp_pace_per_km
    threshold_speed = 3600 / threshold_pace_per_km
    
    intensity_factor = ngp_speed / threshold_speed
    return intensity_factor

# --- rTSS Calculations ---

def calculate_rtss(ngp_pace, duration_seconds, threshold_pace):
    """
    Calculates rTSS using NGP.
    """
    if ngp_pace <= 0 or threshold_pace <= 0: return 0
    
    # Intensity Factor (Inverse because Pace)
    # Threshold 4:00 (240s), NGP 3:00 (180s) -> 240/180 = 1.33 IF
    IF = threshold_pace / ngp_pace
    
    duration_hours = duration_seconds / 3600
    
    # rTSS Formula
    return duration_hours * (IF ** 2) * 100

def calculate_rtss_alternative(duration_seconds, ngp_pace_per_km, threshold_pace_per_km):
    """
    Alternative rTSS calculation using NGP.
    """
    if threshold_pace_per_km <= 0 or ngp_pace_per_km <= 0:
        return 0
    
    intensity_factor = calculate_intensity_factor(ngp_pace_per_km, threshold_pace_per_km)
    
    # (duration * ngp * intensity_factor) / (ftp * 3600) * 100
    rtss = (duration_seconds * ngp_pace_per_km * intensity_factor) / (threshold_pace_per_km * 3600) * 100
    
    return rtss

# --- End rTSS Calculations ---

# --- Fitness and VO2 Model Loaders and Predictors ---

def calculate_avg_watts(act, advanced_stats, basic_stats, velocity_mps, air_density, gravity):
    cadence_sec = act['average_cadence'] / 60.0 # Makes cadence by seconds
    watts = (1.036 * basic_stats['weight'] * velocity_mps) + (basic_stats['weight'] * gravity * advanced_stats['hosc'] * cadence_sec) + (0.5 * air_density * advanced_stats['afrontal'] * 1.4 * velocity_mps**3) + (basic_stats['weight'] * gravity * velocity_mps * act['total_elevation_gain'] / act['distance'])
    return watts

def calculate_minetti_watts(speed_mps, grade_percent, body_mass_kg):
    """Calculates Metabolic Power (Watts) using Minetti (2002)."""
    i = grade_percent / 100.0
    # Minetti Energy Cost (J/kg/m)
    cost_j_kg_m = 155.4*(i**5) - 30.4*(i**4) - 43.3*(i**3) + 46.3*(i**2) + 19.5*i + 3.6
    cost_j_kg_m = max(cost_j_kg_m, 2.0) # Floor to prevent negatives
    
    # Power (W) = Cost (J/kg/m) * Speed (m/s) * Mass (kg)
    return cost_j_kg_m * speed_mps * body_mass_kg

def calculate_tss(watts, ftp, duration_seconds):
    """Calculates Training Stress Score (TSS) from Power."""
    if ftp <= 0 or watts <= 0:
        return 0.0
    # Normalized Power approximation (simplified)
    # Intensity Factor
    adjusted_watts = watts
    if watts > 600 and ftp < 400:
        adjusted_watts = watts * 0.25
        
    IF = adjusted_watts / ftp
    
    # TSS = (sec * watts * IF) / (ftp * 3600) * 100
    return (duration_seconds * adjusted_watts * IF) / (ftp * 3600) * 100

def calculate_gravity(act):
    '''The following is using Newtons Law of Universal Gravitation'''
    # Universal Gravitational Constant (m^3 kg^-1 s^-2)
    G = 6.67430 * (10**-11) 
    # Mass of Earth (kg)
    M = 5.9722 * (10**24)
    # Mean radius of Earth (m)
    R = 6.371 * (10**6)
    r = R + act['elev_high']  # Distance from Earth's center to the object
    gravity = (G * M) / (r**2)
    return gravity

def calculate_air_density(act: Dict[str, Any]) -> float:
    """
    Calculates Air Density using the International Standard Atmosphere (ISA) model.
    Separates 'Standard Temperature' (for pressure) from 'Actual Temperature' (for density).
    """
    elev = act.get('elev_high', 0)

    # Calculate STANDARD Temperature at this altitude (Required for Pressure)
    if elev > TROPOPAUSE_ALT:
        # Stratosphere (simplified constant temp)
        std_temp_at_alt = ISA_SEA_LEVEL_TEMP_K - (LAPSE_RATE * TROPOPAUSE_ALT)
    else:
        std_temp_at_alt = ISA_SEA_LEVEL_TEMP_K - (LAPSE_RATE * elev)
    
    # Calculate Pressure using STANDARD Temperature
    # (Do not use actual temp here, or pressure will be wrong)
    if elev <= TROPOPAUSE_ALT:
        pressure = ISA_SEA_LEVEL_PRESSURE_PA * (std_temp_at_alt / ISA_SEA_LEVEL_TEMP_K) ** EXPONENT
    else:
        # Stratosphere fallback
        p_tropo = ISA_SEA_LEVEL_PRESSURE_PA * ( (ISA_SEA_LEVEL_TEMP_K - (LAPSE_RATE * TROPOPAUSE_ALT)) / ISA_SEA_LEVEL_TEMP_K) ** EXPONENT
        pressure = p_tropo * math.exp(-GRAVITY * (elev - TROPOPAUSE_ALT) / (GAS_CONST * std_temp_at_alt))

    # Use sensor data if available, otherwise fallback to standard
    if act.get('average_temp') is not None:
        actual_temp_k = act['average_temp'] + 273.15
    else:
        actual_temp_k = std_temp_at_alt
        
    # Density depends on the ACTUAL temperature of the gas
    return pressure / (GAS_CONST * actual_temp_k)

def estimate_vdot(distance_meters, time_seconds):
    """
    Estimates VDOT (Effective VO2Max) from run performance.
    Essential for generating training labels for the Neural Net.
    """
    if time_seconds <= 120 or distance_meters <= 0: # Filter short/bad data
        return np.nan
    
    # Velocity in meters/min
    v_min = distance_meters / (time_seconds / 60.0)
    
    # Oxygen Cost (ACSM / Jack Daniels approximation)
    vo2_cost = 0.182258 * v_min + 0.000104 * (v_min**2) - 4.60
    
    # Percent of VO2Max Sustainable for this duration
    time_min = time_seconds / 60.0
    # Regression of world record drops
    percent_max = 0.8 + 0.1894393 * math.exp(-0.012778 * time_min) + 0.2989558 * math.exp(-0.1932605 * time_min)
    
    # VDOT
    return vo2_cost / percent_max

'''
NN Functions start here
'''

def banister_recursive(params, load_history):
    """
    Optimized Banister Model.
    Returns: (fitness, fatigue, prediction)
    """
    k1, k2, p0, tau1, tau2 = params
    
    # Pre-calculate exponential decay factors
    tau1 = max(tau1, 1.0)
    tau2 = max(tau2, 1.0)
    
    d1 = np.exp(-1.0 / tau1)
    d2 = np.exp(-1.0 / tau2)
    
    # y[n] = x[n] + d * y[n-1]
    fitness = lfilter([1], [1, -d1], load_history)
    fatigue = lfilter([1], [1, -d2], load_history)
    
    # Performance = Baseline + Fitness_Gain - Fatigue_Decay
    prediction = p0 + (k1 * fitness) - (k2 * fatigue)
    
    return fitness, fatigue, prediction

def save_model(obj, filename):
    joblib.dump(obj, filename)

def load_model(filename):
    return joblib.load(filename)

def predict_performance_hybrid(load_history, recent_features, banister_params, nn_model):
    """
    Combines Banister Physics + NN Correction.
    """
    _, _, banister_preds = banister_recursive(banister_params, load_history)
    base_performance = banister_preds[-1] # Today's theoretical speed

    correction = nn_model.predict(np.array(recent_features).reshape(1, -1))[0]
    
    return base_performance + correction

# --- End Fitness and VO2 Model Loaders and Predictors ---

# --- Convenience loaders / thin wrappers so other modules can use these models via algorithm.py ---

def load_fitness_nn(path: str = 'fitness_nn_model.pkl'):
    """Load the trained fitness neural net (joblib). Returns estimator."""
    return _joblib.load(path)


def predict_fitness(estimator, features):
    """Run the fitness NN estimator on features (array-like). Returns prediction(s)."""
    return estimator.predict(features)


def load_banister_params(path: str = 'banister_params.pkl'):
    """Load previously saved banister params (numpy array)."""
    return _joblib.load(path)


def load_vo2_pipeline(pkl_path: str = 'vo2_pipeline.pkl') -> Dict[str, Any]:
    """Load VO2 pipeline PKL and return pipeline dict (scaler, label_encoder, model)."""
    return _load_vo2_pipeline(pkl_path)


def predict_vo2(row: Dict[str, Any], pipeline: Dict[str, Any], streams: Dict[str, list] = None) -> float:
    """Convenience wrapper around vo2_model.predict_vo2_from_row."""
    return _predict_vo2_refined(row, pipeline, streams)

# --- End convenience section ---
