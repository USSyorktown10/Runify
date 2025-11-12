import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt

# My directories are cooked so the following makes life easier
try:
    # when imported as package: runify.algorithms.phys_metrics.vo2_nn_trainer
    from .vo2_model import save_pipeline
except Exception:
    # when run as script from the phys_metrics directory: python vo2_nn_trainer.py
    from vo2_model import save_pipeline


def train_and_save(activities_csv: str = 'activities_output.csv', out_dir: str = 'artifacts'):
    """Train VO2 NN on the CSV in the current directory and save artifacts to out_dir.

    Must be run from the `phys_metrics` directory where `activities_output.csv` lives.
    """
    data = pd.read_csv(activities_csv)
    le = LabelEncoder()
    data['gender'] = le.fit_transform(data['gender'].astype(str))  # Binary encoding

    data = data.rename(columns={
        'avg_hr_bpm': 'HR',
        'EE_kcal': 'EE',
        'age': 'age',
        'height_cm': 'height',
        'body_mass_kg': 'body_mass',
        'gender': 'gender'
    })
    X = data[['HR', 'EE', 'age', 'height', 'body_mass', 'gender']]
    y = data['VO2_ml_kg_min']

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.33)

    # Build model
    model = Sequential()
    model.add(Dense(32, activation='elu', input_shape=(X_train.shape[1],)))
    model.add(Dropout(0.2))
    model.add(Dense(16, activation='elu'))
    model.add(Dropout(0.2))
    model.add(Dense(1, activation='linear'))

    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['RootMeanSquaredError'])

    # Fit the model and capture the training history
    history = model.fit(
        X_train, y_train,
        epochs=500,
        batch_size=10,
        validation_split=0.2,
        callbacks=[EarlyStopping(monitor='val_loss', patience=20)],
    )

    # Predictions + scatter
    y_pred = model.predict(X_test)
    # Flatten to 1d
    y_test_arr = np.asarray(y_test).reshape(-1)
    y_pred_arr = np.asarray(y_pred).reshape(-1)

    # Compute evaluation metrics
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    rmse = np.sqrt(mean_squared_error(y_test_arr, y_pred_arr))
    mae = mean_absolute_error(y_test_arr, y_pred_arr)
    r2 = r2_score(y_test_arr, y_pred_arr)
    print(f"Test RMSE: {rmse:.3f}, MAE: {mae:.3f}, R2: {r2:.3f}")
    cmap = plt.get_cmap('viridis')
    colors = cmap((y_test - y_test.min()) / (y_test.max() - y_test.min()))  # Normalize to [0, 1]

    # Create main scatter + fit + residuals subplot
    fig, (ax_scatter, ax_resid) = plt.subplots(2, 1, figsize=(7, 9), gridspec_kw={'height_ratios': [3, 1]}, sharex=True)

    ax_scatter.scatter(y_test_arr, y_pred_arr, c=colors, alpha=0.7)
    ax_scatter.set_xlabel('Actual VO2')
    ax_scatter.set_ylabel('Predicted VO2')
    ax_scatter.set_title('Predicted vs Actual VO2')

    # Identity line
    ax_scatter.plot([y_test_arr.min(), y_test_arr.max()], [y_test_arr.min(), y_test_arr.max()], 'k--', lw=2, label='Identity')

    # Linear regression fit line (trend)
    coeffs = np.polyfit(y_test_arr, y_pred_arr, deg=1)
    poly = np.poly1d(coeffs)
    xs = np.linspace(y_test_arr.min(), y_test_arr.max(), 200)
    ax_scatter.plot(xs, poly(xs), color='red', lw=2, label=f'Fit: y={coeffs[0]:.3f}x+{coeffs[1]:.3f}')

    # Compute correlation
    if len(y_test_arr) > 1:
        corr = np.corrcoef(y_test_arr, y_pred_arr)[0, 1]
    else:
        corr = np.nan

    # Annotate metrics on plot
    n = len(y_test_arr)
    metrics_txt = f"n={n}\nRMSE={rmse:.3f}\nMAE={mae:.3f}\nR2={r2:.3f}\nSlope={coeffs[0]:.3f}\nInt={coeffs[1]:.3f}\nCorr={corr:.3f}"
    ax_scatter.text(0.02, 0.98, metrics_txt, transform=ax_scatter.transAxes, va='top', ha='left', bbox=dict(facecolor='white', alpha=0.7))

    ax_scatter.legend()
    cbar = fig.colorbar(plt.cm.ScalarMappable(cmap=plt.get_cmap('viridis')), ax=ax_scatter)
    cbar.set_label('Actual VO2 Color Scale')

    # Residuals (pred - actual)
    residuals = y_pred_arr - y_test_arr
    ax_resid.scatter(y_test_arr, residuals, c='gray', alpha=0.7)
    ax_resid.axhline(0, color='k', linestyle='--', linewidth=1)
    ax_resid.set_xlabel('Actual VO2')
    ax_resid.set_ylabel('Residual (pred-actual)')

    plt.tight_layout()
    plt.show()

    # Plot training history (loss)
    if 'loss' in history.history:
        plt.figure()
        plt.plot(history.history.get('loss', []), label='train_loss')
        plt.plot(history.history.get('val_loss', []), label='val_loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss (MSE)')
        plt.title('Training history')
        plt.legend()
        plt.show()

    # Save artifacts (PKL + model H5)
    os.makedirs(out_dir, exist_ok=True)
    pkl_path = save_pipeline(scaler, le, model, out_dir=out_dir)
    print(f"Saved pipeline to: {pkl_path}")
    return pkl_path


if __name__ == '__main__':
    # When run from the phys_metrics directory: `python vo2_nn_trainer.py`
    cwd = os.getcwd()
    print(f"Running trainer from: {cwd}")
    train_and_save(activities_csv='activities_output.csv', out_dir='artifacts')
