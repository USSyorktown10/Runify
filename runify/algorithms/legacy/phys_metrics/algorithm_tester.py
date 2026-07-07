import unittest
import numpy as np
import pandas as pd
import joblib
from runify.algorithms.legacy.phys_metrics.algorithm import (
    calculate_gap, 
    calculate_air_density, 
    banister_recursive, 
    calculate_rtss,
    predict_vo2
)

class TestPhysiologicalMetrics(unittest.TestCase):

    def setUp(self):
        # Mock Activity Data (A hard, hilly run at altitude)
        self.hilly_run = {
            'distance': 10.0,      # km
            'elev_high': 2000,     # meters (altitude)
            'total_elevation_gain': 500, # meters
            'average_cadence': 170,
            'average_temp': 15,    # Celsius
            'weight': 70           # kg
        }
        
        # Mock Streams (Time series data)
        # 10 minutes of running
        self.hr_stream = [140] * 100 + [150] * 100 + [145] * 100 # Pretty steady
        self.vel_stream = [3.0] * 100 + [3.2] * 100 + [3.1] * 100 

    def test_air_density_at_altitude(self):
        """Test if air density drops correctly at 2000m."""
        # Sea level density is approx 1.225
        density = calculate_air_density(self.hilly_run)
        print(f"\n[Air Density] At 2000m: {density:.4f} kg/m^3")
        
        self.assertLess(density, 1.225, "Air density should be lower at altitude")
        self.assertGreater(density, 0.9, "Air density shouldn't be too low (not space)")

    def test_gap_calculation(self):
        """Test if Grade Adjusted Pace is faster than actual pace on uphills."""
        pace_per_km = 300  # 5:00/km
        elev_gain = 500    # 500m gain
        dist = 10          # 10km
        grade = elev_gain / dist * 100  # Gradient = 5%
        
        gap = calculate_gap(pace_per_km, grade)
        print(f"[GAP] Raw Pace: {pace_per_km}s/km -> GAP: {gap:.2f}s/km")
        
        # GAP should be LOWER (faster) than raw pace because we worked harder
        self.assertLess(gap, pace_per_km)

    def test_rtss_logic(self):
        """Test rTSS accumulation."""
        duration = 3600 # 1 hour
        ngp = 240       # 4:00/km
        threshold = 240 # 4:00/km (Threshold)
        
        # If running AT threshold for 1 hour, TSS should be exactly 100
        rtss = calculate_rtss(duration, ngp, threshold)
        print(f"[rTSS] 1 hour at Threshold: {rtss:.2f}")
        
        self.assertAlmostEqual(rtss, 100.0, places=1)

    def test_banister_model(self):
        """Test if fitness and fatigue accumulate correctly."""
        # Params: k1=1, k2=2, p0=0, tau1=42, tau2=7
        params = [1.0, 2.0, 0, 42, 7]
        tss_history = np.array([0, 100, 0, 0, 0]) # One hard workout on day 2
        
        fit, fat, pred = banister_recursive(params, tss_history)
        
        print(f"[Banister] Day 2 Fatigue (Should be high): {fat[1]:.2f}")
        print(f"[Banister] Day 5 Fatigue (Should decay): {fat[4]:.2f}")
        
        # Fatigue should spike on day 2 then decay
        self.assertGreater(fat[1], 0)
        self.assertLess(fat[4], fat[1])

    def test_vo2_refined_logic(self):
        """Test the steady-state extraction logic without loading a real ML model."""
        # Mock Pipeline (just a dummy scaler/model to pass the type check)
        class MockModel:
            def predict(self, X): return np.array([55.5]) # Return dummy VO2
            
        class MockScaler:
            def transform(self, X): return X
            
        pipeline = {'model': MockModel(), 'scaler': MockScaler(), 'label_encoder': None}
        
        row = {'age': 30, 'gender': 'male', 'EE': 500}
        streams = {'heartrate': self.hr_stream, 'velocity': self.vel_stream}
        
        # This calls your refined function using the streams
        vo2 = predict_vo2(row, pipeline, streams)
        print(f"[VO2] Predicted from Stream: {vo2}")
        
        self.assertEqual(vo2, 55.5)

if __name__ == '__main__':
    print("=== RUNNING PHYSIOLOGICAL ALGORITHM TESTS ===")
    unittest.main()