import numpy as np
import pandas as pd
from scipy import optimize
from sklearn.neural_network import MLPRegressor
import joblib

# ---------------------------
# Load and parse helpers
# ---------------------------
def parse_elapsed_time(s):
    if isinstance(s, (int, float)):
        return float(s)
    parts = str(s).split(':')
    if len(parts) == 3:
        h, m, sec = parts
        return int(h) * 3600 + int(m) * 60 + float(sec)
    elif len(parts) == 2:
        m, sec = parts
        return int(m) * 60 + float(sec)
    try:
        return float(s)
    except Exception:
        return 0

def parse_numeric(val):
    """Convert to float if possible, NaN if not."""
    try:
        s = str(val).strip()
        if s in ["", "--", "nan", "NaN", "None"]:
            return np.nan
        return float(s.replace(",", ""))  # remove commas if present
    except (ValueError, TypeError):
        return np.nan






# ---------------------------
# Load CSV
# ---------------------------
data = pd.read_csv('algorithms/activities.csv')

# ---------------------------
# Step 1 - Calculate TRIMP-like Load
# ---------------------------
def calculate_trimp(row, hr_rest=60, hr_max=190, gender_const=1.92):
    duration_min = parse_elapsed_time(row['Elapsed Time']) / 60
    if np.isnan(row['Avg HR']) or row['Avg HR'] == 0:
        return np.nan
    hr_ratio = (row['Avg HR'] - hr_rest) / (hr_max - hr_rest)
    return duration_min * hr_ratio * gender_const

data['Load'] = data.apply(calculate_trimp, axis=1)

# Fill missing Load values with pace-based estimate
def calculate_pace_load(row):
    duration_hr = parse_elapsed_time(row['Elapsed Time']) / 3600
    if duration_hr == 0:
        return np.nan
    distance_km = (row['Distance'])  # assuming already in km
    if "," in distance_km:
        distance_km = float(distance_km.replace(",", ""))
        distance_km = distance_km / 1000  # convert to km if in meters
    else:
        distance_km = float(distance_km) * 1.60934 # convert miles to km
    intensity_factor = (distance_km / duration_hr) / 10  # normalize by 10 km/h
    return distance_km * intensity_factor

data['Load'] = data['Load'].fillna(data.apply(calculate_pace_load, axis=1))
data['Load'] = data['Load'].fillna(0)

# ---------------------------
# Step 2 - Target variable: Normalized Speed (m/s)
# ---------------------------
def calculate_speed(row):
    # Get distance as float (meters)
    try:
        dist_str = str(row['Distance']).replace(",", "").strip()
        distance_val = float(dist_str)
    except (ValueError, TypeError):
        return np.nan  # skip if completely invalid

    if distance_val < 100:  # very likely miles
        distance_m = distance_val * 1609.34
    else:  # already in meters
        distance_m = distance_val

    # Parse elapsed time (seconds)
    elapsed_sec = parse_elapsed_time(row['Elapsed Time'])
    if elapsed_sec <= 0:
        return np.nan

    # Speed in m/s
    return distance_m / elapsed_sec

# Fill Speed_mps where missing
data['Speed_mps'] = data.get('Speed_mps', np.nan)  # ensure column exists
data['Speed_mps'] = data['Speed_mps'].fillna(data.apply(calculate_speed, axis=1))
data['Speed_mps'] = data['Speed_mps'].fillna(0)

Actual_Performance = data['Speed_mps'].values


# ---------------------------
# Step 3 - Banister model using new Load
# ---------------------------
TSS = data['Load'].values

def banister_recursive(params, TSS):
    k1, k2, PO, CTLC, ATLC = params
    fitness = np.zeros_like(TSS)
    fatigue = np.zeros_like(TSS)
    for i in range(1, len(TSS)):
        fitness[i] = fitness[i-1] + (TSS[i] - fitness[i-1]) / CTLC
        fatigue[i] = fatigue[i-1] + (TSS[i] - fatigue[i-1]) / ATLC
    prediction = k1 * fitness + k2 * fatigue + PO
    return prediction

def banister_loss(params):
    prediction = banister_recursive(params, TSS)
    return np.mean(np.abs(Actual_Performance - prediction))

initial_guess = [0.1, 0.5, np.mean(Actual_Performance), 45, 15] # type: ignore
result = optimize.minimize(banister_loss, initial_guess)
banister_params = result.x
print("Banister Model Params:", banister_params)

# Save params
joblib.dump(banister_params, 'banister_params.pkl')

# ---------------------------
# Step 4 - Neural net with multiple inputs
# ---------------------------
# Also clean any other key numeric columns used in features
numeric_cols = ['Distance', 'Avg HR', 'Total Ascent', 'Load', 'Speed_mps', 'Avg Run Cadence']
for col in numeric_cols:
    if col in data.columns:
        data[col] = data[col].apply(parse_numeric)
        
window = 28
features = []
targets = []

for i in range(len(data) - window + 1):
    block = data.iloc[i:i+window]
    features.append([
        block['Load'].mean(),
        block['Distance'].mean(),
        block['Avg HR'].mean(),
        block['Avg Run Cadence'].mean(),
        block['Total Ascent'].mean()
    ])
    targets.append(block['Speed_mps'].mean())

features = np.array(features)
targets = np.array(targets)

mask = ~np.isnan(features).any(axis=1) & ~np.isnan(targets)
features_clean = features[mask]
targets_clean = targets[mask]

nn_model = MLPRegressor(solver='lbfgs', activation='relu', hidden_layer_sizes=[50], random_state=42)
nn_model.fit(features_clean, targets_clean)

joblib.dump(nn_model, 'fitness_nn_model.pkl')

print(f"Neural net trained on {len(targets_clean)} samples")
