import joblib
import numpy as np
from runify.algorithms.legacy.phys_metrics.algorithm import estimate_vdot

loaded_data = joblib.load("artifacts/vo2_pipeline.pkl")
model = loaded_data["model"]
scaler = loaded_data["scaler"]
label_encoder = loaded_data["label_encoder"]
raw_categorical_val = 0
raw_numerical_vals = [3.35, 150, 0]
raw_features = np.array([[raw_numerical_vals[0], raw_numerical_vals[1], raw_numerical_vals[2]]])
scaled_features = scaler.transform(raw_features)
prediction = model.predict(scaled_features)
print("VDOT Prediction", prediction[0])

print(estimate_vdot(9656.06, 2882.4))

# Me when realize that train_models.py actually sucks