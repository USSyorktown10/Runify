import os
from typing import Dict, Any

import joblib
import numpy as np
import pandas as pd

# Artifact (old as time)
DEFAULT_ARTIFACT = 'vo2_pipeline.pkl'


def save_pipeline(scaler, label_encoder, model, out_dir: str = '.', artifacts_name: str = DEFAULT_ARTIFACT) -> str:
    """Save scaler, label encoder and sklearn model. Returns path to PKL."""
    os.makedirs(out_dir, exist_ok=True)

    pipeline = {
        'scaler': scaler,
        'label_encoder': label_encoder,
        'model': model,  # sklearn model (no separate .h5 file needed)
    }

    pkl_path = os.path.join(out_dir, artifacts_name)
    joblib.dump(pipeline, pkl_path)
    return pkl_path


def load_pipeline(pkl_path: str) -> Dict[str, Any]:
    """
    Load pipeline PKL with sklearn model.
    """
    if not os.path.exists(pkl_path):
        raise FileNotFoundError(f"Pipeline file not found: {pkl_path}")
        
    pipeline = joblib.load(pkl_path)
    
    # Validate required components
    if 'model' not in pipeline:
        raise ValueError("Pipeline missing 'model' component")
    if 'scaler' not in pipeline:
        raise ValueError("Pipeline missing 'scaler' component")
            
    return pipeline

def extract_steady_state_metrics(hr_stream, velocity_stream, window_size=300):
    """
    Returns mean HR and Speed for the most stable window using integer indexing.
    """
    if len(hr_stream) < window_size:
        return np.mean(hr_stream), np.mean(velocity_stream)

    hr_series = pd.Series(hr_stream)
    # Use rolling standard deviation
    hr_std = hr_series.rolling(window=window_size).std()
    
    # Use np.argmin to get the integer position of the minimum value
    valid_std = hr_std.dropna()
    if valid_std.empty:
        return np.mean(hr_stream), np.mean(velocity_stream)
        
    # Get the integer location of the minimum std deviation
    best_relative_idx = np.argmin(valid_std.values)
    # Add back the window offset because dropna() removed the first (window_size - 1) elements
    best_idx = best_relative_idx + (window_size - 1)
    
    # Slice using integer positions
    steady_hr = np.mean(hr_stream[best_idx - window_size + 1 : best_idx + 1])
    steady_vel = np.mean(velocity_stream[best_idx - window_size + 1 : best_idx + 1])
    
    return steady_hr, steady_vel


def predict_vo2_from_row(row: Dict[str, Any], pipeline: Dict[str, Any]) -> float:
    """Accepts a row-like dict with keys: HR, EE, age, height, body_mass, gender (gender can be str or encoded, pls encode).

    Returns predicted VO2 (ml/kg/min) as float.
    """
    # Build dataframe for consistent ordering
    df = pd.DataFrame([{
        'HR': row.get('HR') or row.get('avg_hr_bpm'),
        'EE': row.get('EE') or row.get('EE_kcal') or row.get('EE_kcals'),
        'age': row.get('age'),
        'height': row.get('height') or row.get('height_cm'),
        'body_mass': row.get('body_mass') or row.get('body_mass_kg'),
        'gender': row.get('gender')
    }])

    le = pipeline.get('label_encoder')
    gender_val = df.loc[0, 'gender']
    
    # Try to encode. If no encoder or failure, force to numeric default.
    if le is not None and isinstance(gender_val, str):
        try:
            df.loc[0, 'gender'] = le.transform([gender_val])[0]
        except Exception:
            df.loc[0, 'gender'] = 0 # Default (Male)
    elif isinstance(gender_val, str):
        # Manual fallback map
        if gender_val.lower() in ['f', 'female', 'woman']:
            df.loc[0, 'gender'] = 1
        else:
            df.loc[0, 'gender'] = 0

    scaler = pipeline['scaler']
    X = df[['HR', 'EE', 'age', 'height', 'body_mass', 'gender']].astype(float)
    X_scaled = scaler.transform(X)

    model = pipeline['model']
    pred = model.predict(X_scaled)
    return float(pred.reshape(-1)[0])

def predict_vo2_refined(row: Dict[str, Any], pipeline: Dict[str, Any], streams: Dict[str, list] = None) -> float:
    """
    Refined VO2 prediction that prioritizes steady-state segments over activity averages.
    """
    # Try to get high-quality steady-state metrics if streams exist
    if streams and 'heartrate' in streams and 'velocity' in streams:
        hr, _ = extract_steady_state_metrics(streams['heartrate'], streams['velocity'])
    else:
        # Fallback to the provided average HR
        hr = row.get('HR') or row.get('avg_hr_bpm') or row.get('average_heartrate')

    # Build the feature dataframe for the model
    df = pd.DataFrame([{
        'HR': hr,
        'EE': row.get('EE') or row.get('EE_kcal') or row.get('kilojoules', 0) * 0.239,
        'age': row.get('age', 30),
        'height': row.get('height') or row.get('height_cm', 175),
        'body_mass': row.get('body_mass') or row.get('body_mass_kg', 70) ,
        'gender': row.get('gender', 0)
    }])

    df['gender'] = df['gender'].astype(object)
    #Force gender to be a number
    le = pipeline.get('label_encoder')
    gender_val = df.loc[0, 'gender']
    
    # Check if it's already a number
    if isinstance(gender_val, (int, float, np.number)):
        pass
    # String + encoder = gender
    elif le is not None and isinstance(gender_val, str):
        try:
            df.loc[0, 'gender'] = le.transform([gender_val])[0]
        except Exception:
            df.loc[0, 'gender'] = 0
    # String + no encoder = fallback
    else:
        g_str = str(gender_val).lower()
        if g_str in ['f', 'female', 'woman', 'w']:
            df.loc[0, 'gender'] = 1 # Female
        else:
            df.loc[0, 'gender'] = 0 # Male (default)
    
    scaler = pipeline['scaler']
    X_scaled = scaler.transform(df[['HR', 'EE', 'age', 'height', 'body_mass', 'gender']].astype(float))
    
    pred = pipeline['model'].predict(X_scaled)
    return float(pred.reshape(-1)[0])
