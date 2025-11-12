import os
import tempfile
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder

from runify.algorithms.phys_metrics import vo2_model, algorithm


class DummyModel:
    def __init__(self, factor=1.0):
        self.factor = factor

    def save(self, path):
        # write a small marker so load_model can detect it
        with open(path, 'w') as f:
            f.write('dummy')

    def predict(self, X):
        # return a simple sum-based prediction for testing
        arr = np.asarray(X)
        return np.array([arr.sum(axis=1) * self.factor])


def save_and_load_pipeline_roundtrip():
    # Prepare scaler and label encoder
    scaler = StandardScaler()
    X = np.array([[1, 2, 3, 4, 5, 0], [2, 3, 4, 5, 6, 1]], dtype=float)
    scaler.fit(X)

    le = LabelEncoder()
    le.fit(['female', 'male'])

    dummy = DummyModel(factor=0.5)

    import tempfile, pathlib
    tmp_path = pathlib.Path(tempfile.mkdtemp())
    out_dir = tmp_path / 'artifacts'
    out_dir.mkdir()

    # Patch keras.load_model used in load_pipeline to return our DummyModel
    original_load = getattr(vo2_model, 'load_model', None)
    try:
        vo2_model.load_model = lambda path: dummy

        # Save pipeline (this will call dummy.save which writes a small file)
        pkl_path = vo2_model.save_pipeline(scaler, le, dummy, out_dir=str(out_dir))
        assert os.path.exists(pkl_path)
        assert os.path.exists(str(out_dir / 'vo2_model.h5'))

        # Load pipeline
        pipeline = vo2_model.load_pipeline(pkl_path)
        assert 'scaler' in pipeline and 'label_encoder' in pipeline and 'model' in pipeline

        # Predict using pipeline via predict_vo2_from_row
        row = {'HR': 1, 'EE': 2, 'age': 3, 'height': 4, 'body_mass': 5, 'gender': 0}
        pred = vo2_model.predict_vo2_from_row(row, pipeline)
        assert isinstance(pred, float)
    finally:
        if original_load is not None:
            vo2_model.load_model = original_load



def algorithm_wrappers_test():
    # Test load_fitness_nn and load_banister_params wrappers
    import tempfile, pathlib
    tmp_path = pathlib.Path(tempfile.mkdtemp())

    dummy_estimator = {'fake': 'est'}
    est_path = tmp_path / 'fitness_nn_model.pkl'
    joblib.dump(dummy_estimator, est_path)

    loaded = algorithm.load_fitness_nn(str(est_path))
    assert loaded == dummy_estimator

    # Save fake banister params
    params = np.array([1.0, 2.0, 3.0, 45.0, 15.0])
    params_path = tmp_path / 'banister_params.pkl'
    joblib.dump(params, params_path)

    loaded_params = algorithm.load_banister_params(str(params_path))
    assert np.allclose(loaded_params, params)

    # For VO2 pipeline, reuse the vo2_model save/load but patch keras load
    scaler = StandardScaler()
    X = np.array([[1, 2, 3, 4, 5, 0], [2, 3, 4, 5, 6, 1]], dtype=float)
    scaler.fit(X)
    le = LabelEncoder(); le.fit(['female', 'male'])
    dummy = DummyModel(factor=2.0)

    art_dir = tmp_path / 'art'
    art_dir.mkdir()
    original_load = getattr(vo2_model, 'load_model', None)
    try:
        vo2_model.load_model = lambda path: dummy

        pkl_path = vo2_model.save_pipeline(scaler, le, dummy, out_dir=str(art_dir))
        pipeline = algorithm.load_vo2_pipeline(pkl_path)

        # ensure predict_vo2 returns a float
        row = {'HR': 1, 'EE': 2, 'age': 3, 'height': 4, 'body_mass': 5, 'gender': 0}
        pred = algorithm.predict_vo2(row, pipeline)
        assert isinstance(pred, float)
    finally:
        if original_load is not None:
            vo2_model.load_model = original_load


def run_all_tests():
    print('Running save/load pipeline roundtrip...')
    save_and_load_pipeline_roundtrip()
    print('Running algorithm wrappers test...')
    algorithm_wrappers_test()
    print('All tests passed')


if __name__ == '__main__':
    run_all_tests()
