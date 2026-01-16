import pandas as pd
import numpy as np
from scipy.optimize import minimize
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
import algorithm as alg

# Personalize to yourself. These are basic configurations
CSV_PATH = 'activities.csv'
USER_FTP = 350
USER_MASS = 61
THRESHOLD_PACE = 201
MIN_HR_FOR_TRAINING = 145

def train_fitness_model():
    df = pd.read_csv(CSV_PATH)
    
    # Sort chronologically
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.sort_values('Date', ascending=True).reset_index(drop=True)

    # Clean Numeric Columns
    print("Cleaning data...")
    cols_to_clean = ['Distance', 'Total Ascent', 'Avg Run Cadence', 'Training Stress Score®', 'Avg HR']
    for c in cols_to_clean:
        if c in df.columns:
            df[c] = df[c].astype(str).str.replace(',', '').replace('--', '0')
            df[c] = pd.to_numeric(df[c], errors='coerce').fillna(0)
    
    # Parse Time
    df['Seconds'] = df['Elapsed Time'].apply(alg.parse_time_to_seconds)

    # Calculate rTSS (better than tss)
    print(f"Calculating rTSS using Threshold Pace: {THRESHOLD_PACE}s/km...")
    
    # If its too long, its prolly meters ngl
    df['Dist_Meters'] = df['Distance'].apply(lambda x: x if x > 100 else x * 1609.34)
    
    # Speed & Grade
    df['Speed_mps'] = df.apply(lambda x: x['Dist_Meters'] / x['Seconds'] if x['Seconds'] > 0 else 0, axis=1)
    df['Grade_Pct'] = df.apply(lambda x: (x['Total Ascent'] / x['Dist_Meters'] * 100) if x['Dist_Meters'] > 0 else 0, axis=1)

    rtss_list = []
    
    for i, row in df.iterrows():
        # Get Summary Metrics
        secs = row['Seconds']
        speed = row['Speed_mps']
        grade = row['Grade_Pct']
        
        if speed > 0 and secs > 0:
            avg_pace_sec_km = 1000.0 / speed
            gap = alg.calculate_gap(avg_pace_sec_km, grade)
            val = alg.calculate_rtss(gap, secs, THRESHOLD_PACE)
            rtss_list.append(val)
        else:
            rtss_list.append(0)

    df['calc_rtss'] = rtss_list
    
    # If it has TSS already, It will take that.
    if 'Training Stress Score®' in df.columns:
         df['Final_Load'] = np.where(df['Training Stress Score®'] > 0, df['Training Stress Score®'], df['calc_rtss'])
    else:
        df['Final_Load'] = df['calc_rtss']

    # Banister Optimization
    print("Optimizing Banister Parameters...")
    load_history = df['Final_Load'].values
    actual_perf = df['Speed_mps'].values 
    hr_data = df['Avg HR'].values
    
    def loss_func(params):
        k1, k2, p0, tau1, tau2 = params
        
        # Physical Validity
        if tau1 <= tau2 + 5: return 99999999
        
        _, _, preds = alg.banister_recursive(params, load_history)
        
        # Training on Quality Runs
        mask = (actual_perf > 2.5) & (hr_data > 140)
        if np.sum(mask) < 5: return 99999999 
        
        return np.mean((preds[mask] - actual_perf[mask])**2)
    
    init_params = [0.1, 0.2, 3.5, 45.0, 10.0]
    
    # Guardrails so its accurate
    bounds = [
        (0.0001, 2.0),  # k1 (Fitness Gain)
        (0.0001, 2.0),  # k2 (Fatigue Cost)
        (1.0, 6.0),     # p0 (Base Speed m/s)
        (40.0, 90.0),   # tau1 (Fitness: Long Term) -> FORCED to be 40+ days
        (5.0, 15.0)     # tau2 (Fatigue: Short Term) -> FORCED to be 5-15 days
    ]
    # Use L-BFGS-B
    res = minimize(loss_func, init_params, method='L-BFGS-B', bounds=bounds)
    best_params = res.x
    print(f"Optimal Params: {best_params}")
    print(f"  - Fitness Tau: {best_params[3]:.1f} days (Target: 42)")
    print(f"  - Fatigue Tau: {best_params[4]:.1f} days (Target: 7)")
    
    alg.save_model(best_params, 'banister_params.pkl')
    
    # Train Neural Net on Residuals
    _, _, physics_preds = alg.banister_recursive(best_params, load_history)
    residuals = actual_perf - physics_preds
    
    X, y = [], []
    window = 7
    for i in range(window, len(df)):
        if actual_perf[i] < 1.0: continue 
        
        win_slice = df.iloc[i-window : i]
        current_hr = df['Avg HR'].iloc[i] if 'Avg HR' in df.columns else 140
        
        feat = [
            win_slice['Final_Load'].mean(),
            win_slice['Total Ascent'].mean(),
            current_hr
        ]
        X.append(feat)
        y.append(residuals[i])
        
    if len(X) > 10:
        # Better R^2 fix
        pipe = make_pipeline(
            StandardScaler(), 
            MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=5000, random_state=42)
        )
        
        pipe.fit(X, y)
        alg.save_model(pipe, 'fitness_nn_model.pkl')
        
        # Calculate R^2 manually since it's a pipeline
        score = pipe.score(X, y)
        print(f"Saved fitness_nn_model.pkl (R^2: {score:.4f})")
    else:
        print("Not enough data to train Neural Net.")

if __name__ == '__main__':
    train_fitness_model()