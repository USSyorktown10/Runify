"""
Model trainer that's simplified, upgraded, and handles data much better from previous versions.
Updated 4.23.26 - NN Model training imporved massivly, switched fitness from MLP to RandomForest, now able to handle multiple users, multiple dates, and more complex data relationships.
"""
import os
import pandas as pd
import numpy as np
import joblib
from datetime import datetime
from typing import Dict, Tuple, Any
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import runify.algorithms.legacy.phys_metrics.algorithm as alg
from sklearn.ensemble import RandomForestRegressor

CSV_PATH = 'activities.csv'

ARTIFACTS_DIR = 'artifacts'
REPORTS_DIR = os.path.join(ARTIFACTS_DIR, 'reports')


def ensure_dirs():
    os.makedirs(ARTIFACTS_DIR, exist_ok=True)
    os.makedirs(REPORTS_DIR, exist_ok=True)


def load_and_clean_data():
    """Load activities CSV."""
    import csv

    # Read CSV so that pandas doesn't get mad
    with open(CSV_PATH, 'r') as f:
        reader = csv.reader(f)
        headers = [h.strip() for h in next(reader)]
        rows = list(reader)
    
    # length stuff
    max_cols = max(len(headers), max(len(r) for r in rows) if rows else 0)
    headers.extend([f'col_{i}' for i in range(len(headers), max_cols)])
    
    rows = [r + [''] * (max_cols - len(r)) for r in rows]
    rows = [r[:max_cols] for r in rows]
    
    df = pd.DataFrame(rows, columns=headers, dtype='object')
    for col in df.columns:
        if df[col].dtype == 'object':
            df[col] = df[col].astype(str).str.strip().str.replace('"', '')
    
    numeric_cols = ['Distance', 'Avg HR', 'Total Ascent', 'Calories', 'Training Stress Score®']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(
                df[col].astype(str).str.replace(',', ''),
                errors='coerce'
            ).fillna(0).astype(float)
    
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        df = df.dropna(subset=['Date'])
        df = df.sort_values('Date', ascending=True).reset_index(drop=True)
    
    if 'Elapsed Time' in df.columns:
        df['Seconds'] = df['Elapsed Time'].apply(alg.parse_time_to_seconds).astype(float)
    else:
        df['Seconds'] = 0
    
    df['Dist_Meters'] = (df.get('Distance', 0).fillna(0) * 1609.34).astype(float)
    
    df['Speed_mps'] = np.where(
        df['Seconds'] > 0,
        df['Dist_Meters'] / df['Seconds'],
        0
    ).astype(float)
    
    df['Grade_Pct'] = np.where(
        df['Dist_Meters'] > 0,
        (df.get('Total Ascent', 0).fillna(0) / df['Dist_Meters']) * 100,
        0
    ).astype(float)
    
    return df

def train_vo2_model(test_size=0.2, cv_folds=5):
    """Train VO2 model."""
    
    df = load_and_clean_data()
    
    df_running = df[
        (df['Activity Type'] == 'Running') &
        (df['Seconds'] > 300) &
        (df['Speed_mps'] > 1.0)
    ].copy()
    
    #debug
    print(f"Running activities: {len(df_running)}")

    # VDOT for VO2
    df_running['VDOT'] = df_running.apply(
        lambda row: alg.estimate_vdot(row['Dist_Meters'], row['Seconds']),
        axis=1
    )
    
    df_running = df_running[(df_running['VDOT'] > 20) & (df_running['VDOT'] < 85)].copy()
    
    print(f"Valid VDOT samples: {len(df_running)}")
    
    if len(df_running) < 10:
        print("Not enough samples for vo2")
        return None, {}
    
    features = df_running[['Speed_mps', 'Avg HR', 'Grade_Pct']].values
    target = df_running['VDOT'].values
    
    X_train, X_test, y_train, y_test = train_test_split(
        features, target, test_size=test_size, random_state=42
    )
    
    print(f"Train samples: {len(X_train)}, Test samples: {len(X_test)}")
    
    # Scale
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train NN
    model = MLPRegressor(
        hidden_layer_sizes=(64, 32),
        max_iter=2000,
        random_state=42,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=20
    )
    model.fit(X_train_scaled, y_train)
    
    # Evaluate
    train_pred = model.predict(X_train_scaled)
    test_pred = model.predict(X_test_scaled)
    
    train_r2 = r2_score(y_train, train_pred)
    test_r2 = r2_score(y_test, test_pred)
    train_rmse = np.sqrt(mean_squared_error(y_train, train_pred))
    test_rmse = np.sqrt(mean_squared_error(y_test, test_pred))
    
    # Cross-validation
    cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=cv_folds, scoring='r2')
    
    # more debug
    print("Vdot tester: Vo2 prediction")
    print(f"Train R^2:   {train_r2:.4f}")
    print(f"Test R^2:    {test_r2:.4f}")
    print(f"Train RMSE: {train_rmse:.2f} ml/kg/min")
    print(f"Test RMSE:  {test_rmse:.2f} ml/kg/min")
    print(f"CV Mean R^2: {cv_scores.mean():.4f} (±{cv_scores.std():.4f})")

    # Create pipeline dict (for compatibility with vo2_model.py)
    label_encoder = LabelEncoder()
    label_encoder.fit([0, 1])
    
    pipeline = {
        'model': model,
        'scaler': scaler,
        'label_encoder': label_encoder
    }
    
    metrics = {
        'train_r2': train_r2,
        'test_r2': test_r2,
        'train_rmse': train_rmse,
        'test_rmse': test_rmse,
        'cv_mean_r2': cv_scores.mean(),
        'cv_std_r2': cv_scores.std(),
        'train_samples': len(X_train),
        'test_samples': len(X_test)
    }
    
    return pipeline, metrics

def vdot_to_threshold_pace(vdot):
    """
    Converts a VDOT value to a Threshold Pace (seconds per kilometer).
    Threshold is defined as 88% of VO2max.
    """
    if vdot <= 0 or np.isnan(vdot):
        return 240.0 # default of 4:00 min/km

    target_vo2 = vdot * 0.88
    a = 0.000104
    b = 0.182258
    c = -(4.60 + target_vo2)
    v_m_min = (-b + np.sqrt(b**2 - 4*a*c)) / (2*a)
    threshold_pace_sec_km = (1000.0 / v_m_min) * 60.0
    
    return threshold_pace_sec_km

def get_user_threshold(df_activities):
    """
    Based on the 95% of the best 20-minute pace or best race-effort VDOT.
    """
    quality_efforts = df_activities[df_activities['IF'] >= 0.85]
    if len(quality_efforts) > 0:
        recent_best_vdot = quality_efforts['VDOT'].max()
    else:
        recent_best_vdot = df_activities['VDOT'].mean() * 1.10
    # Convert VDOT back to a 1-hour sustainable pace
    return vdot_to_threshold_pace(recent_best_vdot)

def train_fitness_model(test_size=0.2, cv_folds=5):
    """Train fitness model (Banister + NN)."""
    df = load_and_clean_data()
    
    # Filter running
    df_running = df[df['Activity Type'] == 'Running'].copy()
    print(f"Running activities: {len(df_running)}")
    def row_to_vdot(row):
        return alg.estimate_vdot(row['Distance'] * 1609.34, row['Seconds'])
    df_running['VDOT'] = df_running.apply(row_to_vdot, axis=1)
    # Calculate GAP and rTSS
    
    X, y = [], []
    all_user_params = {}
    window = 7
    user_configs = {
    '001': {'max_hr': 195}, 
    '002': {'max_hr': 200}
    }
    df_running['User_Max_HR'] = df_running['user_id'].map(lambda x: user_configs.get(x, {}).get('max_hr', 190))
    for user_id, user_df in df_running.groupby('user_id'):
        user_df = user_df.sort_values('Date')
        peak_vdot = peak_vdot = user_df['VDOT'].max() 
        dynamic_threshold = vdot_to_threshold_pace(peak_vdot)
        print(f"\n[User: {user_id}] Threshold: {dynamic_threshold:.1f}s/km | Peak VDOT: {peak_vdot:.1f}")
        print(f"Calibrated Threshold: {dynamic_threshold:.1f}s/km (VDOT: {peak_vdot:.1f})")
        
        def calc_rtss(row):
            if row['Speed_mps'] <= 0 or row['Seconds'] <= 0:
                return pd.Series([0,0], index=['rTSS', 'IF'])
            pace_sec_km = 1000.0 / row['Speed_mps']
            gap = alg.calculate_gap(pace_sec_km, row['Grade_Pct'])
            if_factor = dynamic_threshold / gap
            duration_hours = row['Seconds'] / 3600.0
            rtss = duration_hours * (if_factor**2) * 100
            return pd.Series([rtss, if_factor], index=['rTSS', 'IF'])

        user_df[['rTSS', 'IF']] = user_df.apply(calc_rtss, axis=1)
        user_df['Date'] = pd.to_datetime(user_df['Date']).dt.normalize()
        daily_load = user_df.groupby('Date')['rTSS'].sum()
        full_range = pd.date_range(start=user_df['Date'].min(), end=user_df['Date'].max(), freq='D')
        calendar_df = pd.DataFrame(index=full_range).rename_axis('Date')
        calendar_df = calendar_df.join(daily_load, how='left').fillna(0)
        
        # Optimize Banister params
        from scipy.optimize import minimize

        load_history = calendar_df['rTSS'].values
        actual_perf = user_df.set_index('Date')['Speed_mps']
        hr_data = user_df.set_index('Date')['Avg HR']

        # this is starting to annoy me
        def loss_func(params):
            k1, k2, p0, tau1, tau2 = params
            if tau1 <= tau2 + 5:
                return 99999999
            
            _, _, preds_cal = alg.banister_recursive(params, load_history)
            preds_at_runs = preds_cal[calendar_df.index.get_indexer(user_df['Date'])]
            mask = (actual_perf > 2.5) & (hr_data > 140) # possible error
            if np.sum(mask) < 5:
                return 99999999
            
            return np.mean((preds_at_runs[mask] - actual_perf[mask])**2)
        
        init_params = [0.1, 0.2, 3.5, 45.0, 10.0]
        bounds = [
            (0.0001, 2.0),
            (0.0001, 2.0),
            (1.0, 6.0),
            (40.0, 90.0),
            (5.0, 15.0)
        ]
        
        res = minimize(loss_func, init_params, method='L-BFGS-B', bounds=bounds)
        best_params = res.x
        all_user_params[user_id] = best_params
        
        fitness_cal, fatigue_cal, physics_preds_cal = alg.banister_recursive(best_params, load_history)
        calendar_results = pd.DataFrame({
            'fitness': fitness_cal,
            'fatigue': fatigue_cal,
            'physics_preds': physics_preds_cal
        }, index=calendar_df.index)

        user_df = user_df.merge(calendar_results, left_on='Date', right_index=True)

        print(f"k1 (Fitness Gain): {best_params[0]:.4f}")
        print(f"k2 (Fatigue Cost): {best_params[1]:.4f}")
        print(f"p0 (Base Speed):   {best_params[2]:.2f} m/s")
        print(f"tau1 (Fitness): {best_params[3]:.1f} days")
        print(f"tau2 (Fatigue): {best_params[4]:.1f} days")

        user_df['residual'] = (user_df['Speed_mps'] - user_df['physics_preds']) / user_df['physics_preds']
        user_max_hr = user_df['User_Max_HR'].iloc[0]
        
        from sklearn.pipeline import make_pipeline
        
        for i in range(window, len(user_df)):
            row = user_df.iloc[i]

            if row['Speed_mps'] < 1.0 or row['IF'] < 0.90:
                continue
            
            hr = row['Avg HR'] if not np.isnan(row['Avg HR']) and row['Avg HR'] > 0 else user_df['Avg HR'].mean()
            win_slice = user_df.iloc[i-window : i]
            
            feat = [
                row['fitness'], 
                row['fatigue'], 
                win_slice['rTSS'].mean(),
                win_slice['Total Ascent'].mean(),
                hr / user_max_hr,
                peak_vdot
            ]
            X.append(feat)
            y.append(row['residual'])
        
    X = np.array(X)
    y = np.array(y)
    
    #debug
    print(f"\nNN training samples: {len(X)}")
    
    if len(X) < 10:
        print("Not enough residuals to train NN")
        return best_params, None, {}
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
    
    pipe = make_pipeline(
    StandardScaler(),
    RandomForestRegressor(
        n_estimators=100, 
        max_depth=4,
        min_samples_leaf=10,
        random_state=42
        )
    )
    
    pipe.fit(X_train, y_train)
    
    train_r2 = pipe.score(X_train, y_train)
    test_r2 = pipe.score(X_test, y_test)
    
    cv_scores = cross_val_score(pipe, X_train, y_train, cv=cv_folds, scoring='r2')
    
    print(f"Train R^2:   {train_r2:.4f}")
    print(f"Test R^2:    {test_r2:.4f}")
    print(f"CV Mean R^2: {cv_scores.mean():.4f} (±{cv_scores.std():.4f})")

    metrics = {
        'nn_train_r2': train_r2,
        'nn_test_r2': test_r2,
        'nn_cv_mean_r2': cv_scores.mean(),
        'nn_cv_std_r2': cv_scores.std(),
        'nn_train_samples': len(X_train),
        'nn_test_samples': len(X_test)
    }
    
    return best_params, pipe, metrics


def save_all(vo2_pipeline, vo2_metrics, banister_params, fitness_nn, fitness_metrics):
    """Save models, make a report"""
    ensure_dirs()
    
    if vo2_pipeline:
        path = os.path.join(ARTIFACTS_DIR, 'vo2_pipeline.pkl')
        joblib.dump(vo2_pipeline, path)
        print(f"\nSaved VO2 pipeline to {path}")
    
    if banister_params is not None:
        path = os.path.join(ARTIFACTS_DIR, 'banister_params.pkl')
        joblib.dump(banister_params, path)
        print(f"Saved Banister params to {path}")
    
    if fitness_nn:
        path = os.path.join(ARTIFACTS_DIR, 'fitness_nn_model.pkl')
        joblib.dump(fitness_nn, path)
        print(f"Saved Fitness NN to {path}")
    
    # Save report
    report_path = os.path.join(REPORTS_DIR, f'evaluation_{datetime.now().strftime("%Y%m%d_%H%M%S")}.txt')
    
    with open(report_path, 'w') as f:
        f.write(f"Report from testing at: {datetime.now().isoformat()}\n")
        
        if vo2_metrics:
            f.write("VO2 MODEL\n")
            f.write(f"Train R^2: {vo2_metrics.get('train_r2', 'N/A'):.4f}\n")
            f.write(f"Test R^2:  {vo2_metrics.get('test_r2', 'N/A'):.4f}\n")
            f.write(f"Train RMSE: {vo2_metrics.get('train_rmse', 'N/A'):.2f}\n")
            f.write(f"Test RMSE:  {vo2_metrics.get('test_rmse', 'N/A'):.2f}\n")
            f.write(f"CV Mean R^2: {vo2_metrics.get('cv_mean_r2', 'N/A'):.4f}\n")
            f.write(f"Training samples: {vo2_metrics.get('train_samples', 0)}\n")
            f.write(f"Test samples: {vo2_metrics.get('test_samples', 0)}\n\n")
        
        if fitness_metrics:
            f.write("FITNESS MODEL\n")
            f.write(f"NN Train R^2: {fitness_metrics.get('nn_train_r2', 'N/A'):.4f}\n")
            f.write(f"NN Test R^2:  {fitness_metrics.get('nn_test_r2', 'N/A'):.4f}\n")
            f.write(f"NN CV Mean R^2: {fitness_metrics.get('nn_cv_mean_r2', 'N/A'):.4f}\n")

    print(f"Saved report to {report_path}")


if __name__ == '__main__':    
    vo2_pipeline, vo2_metrics = train_vo2_model()
    banister_params, fitness_nn, fitness_metrics = train_fitness_model()
    
    save_all(vo2_pipeline, vo2_metrics, banister_params, fitness_nn, fitness_metrics)
    
    print("\nComplete :)")
