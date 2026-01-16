import math
import joblib
from datetime import datetime, timezone
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import List

@dataclass
class UserFitness:
    id: int  
    user_id: int
    activity_id: int
    day: str  
    watts: float
    relative_effort: float
    CTL: float
    ATL: float
    norm_CTL: float
    norm_ATL: float
    TSB: float
    
@dataclass
class ActivityFitnessChanges:
    relative_effort: float
    ctl_change: float
    atl_change: float
    watts: float
    tsb_change: float


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

def banister_recursive(params, tss_list):
    k1, k2, PO, CTLC, ATLC = params
    fitness = np.zeros(len(tss_list))
    fatigue = np.zeros(len(tss_list))
    for i in range(1, len(tss_list)):
        fitness[i] = fitness[i-1] + (tss_list[i] - fitness[i-1]) / CTLC
        fatigue[i] = fatigue[i-1] + (tss_list[i] - fatigue[i-1]) / ATLC
    prediction = k1 * fitness + k2 * fatigue + PO
    return fitness, fatigue, prediction


def calculate_tss(activity, advanced_stats, basic_stats, watts):
    duration_hr = activity['elapsed_time'] / 3600
    # 1. Heart Rate
    if activity.get('average_heartrate') and advanced_stats['max_hr'] and advanced_stats['resting_hr']:
        hr_avg = activity['average_heartrate']
        hr_rest = advanced_stats['resting_hr']
        hr_max = advanced_stats['max_hr']
        if hr_max <= hr_rest:
            return duration_hr * 50  # fallback
        intensity = (hr_avg - hr_rest) / (hr_max - hr_rest)
    # 2. RPE
    elif activity.get('rpe'):
        rpe = float(activity.get('rpe', 5))
        intensity = rpe / 10  # Assuming 1–10 RPE scale
    # 3. Fastest pace
    elif basic_stats['fastest_mile'] and activity.get('distance') and activity.get('elapsed_time'):
        pace = activity['distance'] / activity['elapsed_time']  # m/s
        intensity = pace / basic_stats['fastest_mile']
    # 4. FTP
    elif advanced_stats['ftp'] and watts:
        intensity = watts / advanced_stats['ftp']
    else:
        intensity = 0.5  # fallback

    intensity = max(0, min(intensity, 1))
    tss = duration_hr * intensity * 100

    # Apply terrain gradient
    if activity.get('total_elevation_gain', 0) > 0 and activity.get('distance', 0) > 0:
        gradient = activity['total_elevation_gain'] / activity['distance']
        tss *= (1 + gradient * 0.1)
    return tss

def calculate_air_density(act):
    '''The following is using International Standard Atmosphere (ISA) model as a basis'''        
    try:
        # Temperature calculation
        if act.get('average_temp'):
            if act['average_temp'] is None:
                T = ISA_SEA_LEVEL_TEMP_C - LAPSE_RATE * act['elev_high'] if act['elev_high'] <= TROPOPAUSE_ALT else TROPOPAUSE_TEMP_C
            else:
                T = act['average_temp']
            T_kelvin = T + 273.15

            # Precompute constants
            sea_level_temp_K = ISA_SEA_LEVEL_TEMP_C + 273.15
            exponent = GRAVITY / (LAPSE_RATE * GAS_CONST)

            # Pressure calculation
            if act['elev_high'] <= TROPOPAUSE_ALT:
                P = ISA_SEA_LEVEL_PRESSURE_PA * (T_kelvin / sea_level_temp_K) ** exponent
            else:
                tropopause_temp_K = TROPOPAUSE_TEMP_C + 273.15
                P_tropopause = ISA_SEA_LEVEL_PRESSURE_PA * (tropopause_temp_K / sea_level_temp_K) ** exponent
                P = P_tropopause * math.exp(-GRAVITY * (act['elev_high'] - TROPOPAUSE_ALT) / (GAS_CONST * T_kelvin))

            # Air density calculation
            air_density = P / (GAS_CONST * T_kelvin)
        else:
            T = ISA_SEA_LEVEL_TEMP_C - LAPSE_RATE * act['elev_high'] if act['elev_high'] <= TROPOPAUSE_ALT else TROPOPAUSE_TEMP_C
            T_kelvin = T + 273.15

            # Precompute constants
            sea_level_temp_K = ISA_SEA_LEVEL_TEMP_C + 273.15
            exponent = GRAVITY / (LAPSE_RATE * GAS_CONST)

            # Pressure calculation
            if act['elev_high'] <= TROPOPAUSE_ALT:
                # Assuming a standard lapse rate (temperature decrease with altitude)
                lapse_rate = 0.0065  # K/m (approximate lapse rate in the troposphere)
                estimated_T_kelvin = ISA_SEA_LEVEL_TEMP_K - (lapse_rate * act['elev_high'])
                P = ISA_SEA_LEVEL_PRESSURE_PA * (estimated_T_kelvin / ISA_SEA_LEVEL_TEMP_K) ** EXPONENT
            else:
                # For altitudes above the tropopause, use tropopause values and an exponential decay
                tropopause_temp_K = TROPOPAUSE_TEMP_C + 273.15
                P_tropopause = ISA_SEA_LEVEL_PRESSURE_PA * (tropopause_temp_K / ISA_SEA_LEVEL_TEMP_K) ** EXPONENT
                P = P_tropopause * math.exp(-GRAVITY * (act['elev_high'] - TROPOPAUSE_ALT) / (GAS_CONST * tropopause_temp_K)) # Use tropopause temp for this part of the calculation, assuming a constant temperature in the stratosphere

            # Air density calculation
            air_density = P / (GAS_CONST * T_kelvin)
            return air_density
    except Exception as e:
        return e
    
def calculate_avg_watts(act, advanced_stats, basic_stats, velocity_mps, air_density, gravity):
    cadence_sec = act['average_cadence'] / 60.0
    watts = (1.036 * basic_stats['weight'] * velocity_mps) + (basic_stats['weight'] * gravity * advanced_stats['hosc'] * cadence_sec) + (0.5 * air_density * advanced_stats['afrontal'] * 1.4 * velocity_mps**3) + (basic_stats['weight'] * gravity * velocity_mps * act['total_elevation_gain'] / act['distance'])
    return watts

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

def get_user_fitness(activities, advanced_stats, basic_stats, banister_params=joblib.load('banister_params.pkl'), current_user=None): 
    now = datetime.now(timezone.utc)
    fitness_records: List[UserFitness] = []
    
    # Group activities by day
    from collections import defaultdict
    acts_by_day = defaultdict(list)
    for activity in activities:
        day = activity['start_date_local'][:10]
        acts_by_day[day].append(activity)

    sorted_days = sorted(acts_by_day.keys())
    CTL = 0
    ATL = 0
    ctl_by_day = {}
    atl_by_day = {}
    daily_data = {}
    re_list = []

    for day in sorted_days:
        re_today = 0
        for act in acts_by_day[day]:
            if not act.get('distance') or not act.get('elapsed_time') or not act.get('average_cadence'):
                act['relative_effort'] = 0
                continue

            activity_timestamp = datetime.fromisoformat(act['start_date_local'].replace('Z', '+00:00'))
            if activity_timestamp.tzinfo is None:
                activity_timestamp = activity_timestamp.replace(tzinfo=timezone.utc)
            else:
                activity_timestamp = activity_timestamp.astimezone(timezone.utc)
            day = activity_timestamp.date().isoformat()
            velocity_mps = act['distance'] / act['elapsed_time']

            air_density = calculate_air_density(act)
            gravity = calculate_gravity(act)
            
            watts = calculate_avg_watts(act, advanced_stats, basic_stats, velocity_mps, air_density, gravity)
            
            act['watts'] = watts
            act['relative_effort'] = calculate_tss(act, advanced_stats, basic_stats, watts)
            re_today += act['relative_effort'] 
            re_list.append(act['relative_effort'])
        
        CTL_arr, ATL_arr, _ = banister_recursive(banister_params, re_list)
        CTL = CTL + (re_today - CTL) / Tfit
        ATL = ATL + (re_today - ATL) / Tfat
        ctl_by_day[day] = CTL
        atl_by_day[day] = ATL
        for act in acts_by_day[day]:
            key = (day, act['id'])
            daily_data[key] = {
                'watts': watts,
                'relative_effort': act['relative_effort'],
                'ctl': CTL,
                'atl': ATL,
                'tsb': CTL - ATL
            }

    for i, ((day, activity_id), data) in enumerate(list(daily_data.items())[:len(CTL_arr)]):
        ctl = CTL_arr[i]
        atl = ATL_arr[i]
        tsb = ctl - atl
        norm_ctl = (ctl - CTLmin) / (CTLmax - CTLmin) * 100 if ctl > 0 else 0
        norm_atl = (atl - ATLmin) / (ATLmax - ATLmin) * 100 if atl > 0 else 0

        # Create a dataclass instance and append it to the list
        fitness_records.append(
            UserFitness(
                id=activity_id,
                user_id=current_user.id,
                activity_id=activity_id,
                day=day,
                watts=data['watts'],
                relative_effort=data['relative_effort'],
                CTL=ctl,
                ATL=atl,
                norm_CTL=norm_ctl,
                norm_ATL=norm_atl,
                TSB=tsb,
            )
        )

    return fitness_records

def get_activity_fitness_changes(activity_id, activity_data, previous_data):
    if not activity_data:
        return None

    activity_day, watts, relative_effort, ctl, atl, tsb = activity_data
    # Find the previous day's fitness data
    activity_date = datetime.strptime(activity_day, '%Y-%m-%d').date()
    previous_date = activity_date.fromordinal(activity_date.toordinal() - 1)
    previous_day_str = previous_date.isoformat()

    prev_ctl, prev_atl, prev_tsb = (previous_data if previous_data and all(v is not None for v in previous_data) else (0.0, 0.0, 0.0)) # Added type hint for floats


    # Create and return a dataclass instance
    return ActivityFitnessChanges(
        relative_effort=relative_effort,
        ctl_change=ctl - prev_ctl,
        atl_change=atl - prev_atl,
        watts=watts,
        tsb_change=tsb - prev_tsb
    )