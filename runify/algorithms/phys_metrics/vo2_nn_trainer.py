import pandas as pd
import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import algorithm as alg

# CONFIG
CSV_PATH = 'activities.csv'
FTP_GUESS = 350 # Used for relative intensity if needed
BODY_MASS_KG = 61 # Default if not found

def train_vo2_model():
    # Load & Clean Data
    df = pd.read_csv(CSV_PATH)
    
    # Fix numeric columns (remove commas, handle '--')
    cols_to_clean = ['Distance', 'Avg HR', 'Total Ascent', 'Avg Run Cadence', 'Calories']
    for col in cols_to_clean:
        if col in df.columns:
            df[col] = df[col].astype(str).str.replace(',', '').replace('--', '0')
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
            
    # Parse Time
    df['Seconds'] = df['Elapsed Time'].apply(alg.parse_time_to_seconds)
    
    # Miles to meters
    df['Dist_Meters'] = df['Distance'] * 1609.34 
    
    # Calculate Speed (m/s)
    df['Speed_mps'] = df['Dist_Meters'] / df['Seconds']
    
    # Filter for valid running data (Running only, > 5 mins, > 0 speed)
    df = df[ (df['Activity Type'] == 'Running') & (df['Seconds'] > 300) & (df['Speed_mps'] > 1.0) ].copy()
    
    # VDOT yay
    df['VDOT_Label'] = df.apply(lambda row: alg.estimate_vdot(row['Dist_Meters'], row['Seconds']), axis=1)
    
    # Filter reasonable VDOTs (20-85)
    df = df[ (df['VDOT_Label'] > 20) & (df['VDOT_Label'] < 85) ]

    print(f"Training on {len(df)} runs with valid VDOT data.")

    # Get inputs
    df['Grade_Pct'] = (df['Total Ascent'] / df['Dist_Meters']) * 100
    df['Grade_Pct'] = df['Grade_Pct'].fillna(0)
    
    features = df[['Speed_mps', 'Avg HR', 'Grade_Pct']].values
    target = df['VDOT_Label'].values
    
    # Train Neural Net
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(features)
    
    # MLP: 2 hidden layers
    nn = MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=2000, random_state=42)
    nn.fit(X_scaled, target)
    
    score = nn.score(X_scaled, target)
    print(f"Model R^2 Score: {score:.4f}")
    
    # Save Pipeline
    pipeline = {'model': nn, 'scaler': scaler}
    alg.save_model(pipeline, 'vo2_pipeline.pkl')
    print("Saved vo2_pipeline.pkl")

if __name__ == '__main__':
    train_vo2_model()