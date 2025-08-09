import numpy as np
import pandas as pd
from scipy import optimize
from sklearn.neural_network import MLPRegressor
import joblib

# Load your data
data = pd.read_csv('algorithms/activities.csv')

def parse_elapsed_time(s):
    # Handles "hh:mm:ss" or "mm:ss" or seconds as string/int
    if isinstance(s, (int, float)):
        return float(s)
    parts = str(s).split(':')
    if len(parts) == 3:
        h, m, sec = parts
        return int(h) * 3600 + int(m) * 60 + float(sec)
    elif len(parts) == 2:
        m, sec = parts
        return int(m) * 60 + float(sec)
    else:
        try:
            return float(s)
        except Exception:
            return 0

def parse_pace(pace_str):
    # Handles "mm:ss" or "m:ss" or "h:mm:ss"
    if isinstance(pace_str, (int, float)):
        return float(pace_str)
    parts = str(pace_str).split(':')
    if len(parts) == 3:
        h, m, s = parts
        return int(h) * 3600 + int(m) * 60 + float(s)
    elif len(parts) == 2:
        m, s = parts
        return int(m) * 60 + float(s)
    else:
        try:
            return float(pace_str)
        except Exception:
            return np.nan

def calculate_tss(row):
    # Example: use duration (in seconds) and RPE (1-10 scale)
    duration_hr = parse_elapsed_time(row['Elapsed Time']) / 3600
    rpe = row.get('RPE', 5)  # Default to 5 if not present
    intensity = float(rpe) / 10
    tss = duration_hr * intensity * 100
    return tss

data = pd.read_csv('algorithms/activities.csv')
if 'RPE' not in data.columns:
    data['RPE'] = 5  # or set this from your DB if you have it

data['TSS_calc'] = data.apply(calculate_tss, axis=1)
data.to_csv('algorithms/activities_with_TSS.csv', index=False)

data['Avg Pace (sec)'] = data['Avg Pace'].apply(parse_pace)
Actual_Performance = data['Avg Pace (sec)'].values
TSS = data['Training Stress Score®'].values

# Banister recursive model
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

initial_guess = [0.1, 0.5, 50, 45, 15]
result = optimize.minimize(banister_loss, initial_guess)
banister_params = result.x
print("Banister Model Params:", banister_params)
joblib.dump(banister_params, 'banister_params.pkl')

# Neural net (optional, for performance prediction)
window = 28
Offset_Performance = [np.mean(Actual_Performance[i:i+window]) for i in range(len(Actual_Performance)-window+1)]
Block_TSS = [np.mean(TSS[i:i+window]) for i in range(len(TSS)-window+1)]
Block_TSS_np = np.array(Block_TSS).reshape(-1, 1)
Offset_Performance_np = np.array(Offset_Performance)
nn_model = MLPRegressor(solver='lbfgs', activation='relu', hidden_layer_sizes=[50], random_state=42)

# Remove NaNs from Block_TSS and Offset_Performance
mask = ~(
    np.isnan(Block_TSS_np.flatten()) | 
    np.isnan(Offset_Performance_np)
)
Block_TSS_np_clean = Block_TSS_np[mask]
Offset_Performance_np_clean = Offset_Performance_np[mask]

# Reshape Block_TSS_np_clean for sklearn
Block_TSS_np_clean = Block_TSS_np_clean.reshape(-1, 1)

nn_model.fit(Block_TSS_np_clean, Offset_Performance_np_clean)

joblib.dump(banister_params, 'banister_params.pkl')
print(f"Training neural net on {len(Offset_Performance_np_clean)} samples (removed {len(Offset_Performance_np) - len(Offset_Performance_np_clean)} NaN rows)")
print(data[pd.isna(data['Avg Pace (sec)'])])
print("Neural net trained and saved.")