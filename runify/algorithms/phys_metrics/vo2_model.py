import os
from typing import Dict, Any

import joblib
import numpy as np
import pandas as pd
from keras.models import load_model

# Artifact (old as time)
DEFAULT_ARTIFACT = 'vo2_pipeline.pkl'


def save_pipeline(scaler, label_encoder, model, out_dir: str = '.', artifacts_name: str = DEFAULT_ARTIFACT) -> str:
    """Save scaler, label encoder and Keras model. Returns path to PKL."""
    os.makedirs(out_dir, exist_ok=True)
    model_path = os.path.join(out_dir, 'vo2_model.h5')
    model.save(model_path)

    pipeline = {
        'scaler': scaler,
        'label_encoder': label_encoder,
        'model_path': model_path,
    }

    pkl_path = os.path.join(out_dir, artifacts_name)
    joblib.dump(pipeline, pkl_path)
    return pkl_path


def load_pipeline(pkl_path: str) -> Dict[str, Any]:
    """Load pipeline PKL and Keras model. Returns dict with scaler, label_encoder, model."""
    pipeline = joblib.load(pkl_path)
    model = load_model(pipeline['model_path'])
    pipeline['model'] = model
    return pipeline


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
    if le is not None and df.loc[0, 'gender'] is not None and isinstance(df.loc[0, 'gender'], str):
        try:
            df.loc[0, 'gender'] = le.transform([df.loc[0, 'gender']])[0]
        except Exception:
            # If unknown gender string (like cardboard box), fall back to 0
            df.loc[0, 'gender'] = 0

    scaler = pipeline['scaler']
    X = df[['HR', 'EE', 'age', 'height', 'body_mass', 'gender']].astype(float)
    X_scaled = scaler.transform(X)

    model = pipeline['model']
    pred = model.predict(X_scaled)
    return float(pred.reshape(-1)[0])
