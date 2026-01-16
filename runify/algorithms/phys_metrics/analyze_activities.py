import pandas as pd
import numpy as np
import os
import joblib
import algorithm as alg

# CONFIG
INPUT_CSV = 'activities.csv'
OUTPUT_CSV = 'athlete_profile.csv'
BANISTER_PKL = 'banister_params.pkl'
VO2_PKL = 'vo2_pipeline.pkl'
FITNESS_NN_PKL = 'fitness_nn_model.pkl'

# MUST MATCH TRAINER
THRESHOLD_PACE = 201 

# robust is a cool word
def load_robust_pipeline(pkl_path):
    """
    Smart Loader: Handles both raw Pipeline objects and Dictionaries.
    """
    if not os.path.exists(pkl_path):
        raise FileNotFoundError(f"{pkl_path} not found.")
        
    loaded = joblib.load(pkl_path)
    
    # If its a dict
    if isinstance(loaded, dict):
        if 'pipeline' in loaded: return loaded['pipeline']
        if 'model' in loaded: return loaded['model']
        return loaded
        
    #If its an object already
    return loaded

def main():
    if not os.path.exists(INPUT_CSV):
        print(f"Error: {INPUT_CSV} not found.")
        return
    df = pd.read_csv(INPUT_CSV)
    
    if 'Date' in df.columns:
        df['Date_Obj'] = pd.to_datetime(df['Date'], errors='coerce')
        df = df.sort_values('Date_Obj').reset_index(drop=True)

    # Clean Data
    print("[1/4] Processing Data...")
    cols_to_clean = ['Distance', 'Total Ascent', 'Avg Run Cadence', 'Training Stress Score®', 'Avg HR', 'Calories']
    for c in cols_to_clean:
        if c in df.columns:
            df[c] = df[c].astype(str).str.replace(',', '').replace('--', '0')
            df[c] = pd.to_numeric(df[c], errors='coerce').fillna(0)
            
    df['Seconds'] = df['Elapsed Time'].apply(alg.parse_time_to_seconds)
    
    # Distance Fix
    df['Dist_Meters'] = df['Distance'].apply(lambda x: x if x > 100 else x * 1609.34)
    df['Speed_mps'] = df.apply(lambda x: x['Dist_Meters'] / x['Seconds'] if x['Seconds'] > 0 else 0, axis=1)
    df['Grade_Pct'] = df.apply(lambda x: (x['Total Ascent'] / x['Dist_Meters'] * 100) if x['Dist_Meters'] > 0 else 0, axis=1)
    
    # Calculate rTSS
    rtss_vals = []
    for i, row in df.iterrows():
        secs = row['Seconds']
        speed = row['Speed_mps']
        grade = row['Grade_Pct']
        if speed > 0 and secs > 0:
            avg_pace = 1000.0 / speed
            gap = alg.calculate_gap(avg_pace, grade)
            rtss = alg.calculate_rtss(gap, secs, THRESHOLD_PACE)
            rtss_vals.append(rtss)
        else:
            rtss_vals.append(0)
    df['calc_rtss'] = rtss_vals
    
    if 'Training Stress Score®' in df.columns:
        df['Final_Load'] = np.where(df['Training Stress Score®'] > 0, df['Training Stress Score®'], df['calc_rtss'])
    else:
        df['Final_Load'] = df['calc_rtss']

    # Physics (banister)
    print("[2/4] Applying Physics Engine...")
    try:
        params = alg.load_banister_params(BANISTER_PKL)
        ctl, atl, pred_base = alg.banister_recursive(params, df['Final_Load'].values)
        
        # Normalize Display
        fitness_tau = params[3]
        fatigue_tau = params[4]
        scale_fit = (1 - np.exp(-1/fitness_tau))
        scale_fat = (1 - np.exp(-1/fatigue_tau))
        
        df['CTL_Fitness'] = ctl * scale_fit
        df['ATL_Fatigue'] = atl * scale_fat
        df['TSB_Form'] = df['CTL_Fitness'] - df['ATL_Fatigue']
        df['Physics_Speed'] = pred_base
    except Exception as e:
        print(f"Banister failed: {e}")

    # Apply NN
    print("[3/4] Applying Fitness AI...")
    try:
        fitness_pipe = alg.load_fitness_nn(FITNESS_NN_PKL)
        window = 7
        corrections = [0.0] * len(df)
        
        for i in range(window, len(df)):
            if df['Speed_mps'].iloc[i] < 0.1: continue
            
            sl = df.iloc[i-window:i]
            f_load = sl['Final_Load'].mean()
            f_asc = sl['Total Ascent'].mean()
            f_hr = df['Avg HR'].iloc[i]
            
            # Use dataframe to match feature names if pipeline expects it
            feat_df = pd.DataFrame([[f_load, f_asc, f_hr]], columns=['Final_Load', 'Total Ascent', 'Avg HR'])
            
            # Predict (handle both pipeline and raw model)
            try:
                raw_correction = fitness_pipe.predict(feat_df)[0]
            except:
                # Fallback for array input
                raw_correction = fitness_pipe.predict(np.array([[f_load, f_asc, f_hr]]))[0]
            
            # Safety Clamp
            safe_correction = max(-1.5, min(1.5, raw_correction))
            corrections[i] = safe_correction

        df['AI_Correction'] = corrections
        df['Predicted_Speed'] = df['Physics_Speed'] + df['AI_Correction']
        df['Predicted_Speed'] = df['Predicted_Speed'].clip(lower=1.0)
        df['Pred_Pace_MinKm'] = df['Predicted_Speed'].apply(lambda s: (1000/s)/60 if s > 0 else 0)
        
    except Exception as e:
        print(f"Fitness NN skipped: {e}")

    # Vo2 max NN (kinda dumb rn)
    print("[4/4] Applying VO2 Max AI...")
    try:
        # Smart loader
        vo2_pipe = load_robust_pipeline(VO2_PKL)
        
        vo2_estimates = []
        
        for i, row in df.iterrows():
            if row['Speed_mps'] > 2.0 and row['Seconds'] > 300:
                features = pd.DataFrame([[
                    row['Speed_mps'], 
                    row['Avg HR'], 
                    row['Grade_Pct']
                ]], columns=['Speed_mps', 'Avg HR', 'Grade_Pct'])
                
                try:
                    # Lowkey gotta define predict but dont feel like it rn
                    pred_vo2 = vo2_pipe.predict(features)[0]
                except AttributeError:
                    print("Debug: VO2 Pipe object has no predict method.")
                    pred_vo2 = np.nan
                
                vo2_estimates.append(pred_vo2)
            else:
                vo2_estimates.append(np.nan)
        
        df['VO2_Max_Est'] = vo2_estimates
        df['VO2_Trend'] = df['VO2_Max_Est'].rolling(window=5, min_periods=1).mean()
        
    except Exception as e:
        print(f"VO2 AI skipped: {e}")

    # Export
    out_cols = [
        'Date', 'Distance', 'Seconds', 'Final_Load', 
        'CTL_Fitness', 'ATL_Fatigue', 'TSB_Form', 
        'Pred_Pace_MinKm', 'VO2_Max_Est', 'VO2_Trend'
    ]
    
    df[[c for c in out_cols if c in df.columns]].to_csv(OUTPUT_CSV, index=False)
    print(f"Saved to {OUTPUT_CSV}")
    
    print("\n=== LATEST STATS ===")
    # SAFE PRINT: Only print columns that actually exist
    available_cols = [c for c in ['Date', 'CTL_Fitness', 'TSB_Form', 'Pred_Pace_MinKm', 'VO2_Trend'] if c in df.columns]
    print(df[available_cols].tail(1).to_string(index=False))

if __name__ == "__main__":
    main()