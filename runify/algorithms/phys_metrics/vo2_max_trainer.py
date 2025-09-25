import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd
from tensorflow import keras

# LCARS || Data Intake (from Strava Stream JSON/CSV)
# Replace 'run_data.csv' with your extracted Strava stream data
raw_data = pd.read_csv('activities.csv')
features = ['distance', 'elapsed_time', 'total_elevation_gain', 'average_heartrate', 'max_heartrate', 'average_cadence']  # Adjust as available
X = raw_data[features].fillna(0).values  # Input features
y = raw_data['vo2max_lab'].values        # Target (actual VO2 MAX), must be provided for trainer

# LCARS || Data Normalization
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# LCARS || Neural Network Construction
model = keras.Sequential([
    keras.layers.InputLayer(input_shape=(len(features),)),
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dense(32, activation='relu'),
    keras.layers.Dense(1)  # Regression to VO2 MAX (mL/kg/min)
])
model.compile(optimizer='adam', loss='mse', metrics=['mae'])

# LCARS || Trainer Routine
model.fit(X_train, y_train, epochs=100, batch_size=16, validation_data=(X_test, y_test))

# LCARS || Prediction (Estimation from new Strava streams)
predicted_vo2max = model.predict(X_test)
