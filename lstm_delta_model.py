# ==========================================================
# LSTM v2: predict DAY-OVER-DAY ALTITUDE CHANGE (delta), not
# absolute altitude. Reconstruct absolute altitude as
#   pred_altitude[t] = actual_altitude[t-1] + pred_delta[t]
# so the model is anchored to the true previous value the
# same way the persistence baseline is - Kp only has to
# explain the residual on top of that.
#
# Run this in Google Colab. Upload iss_25544_2023.csv (your
# Space-Track gp_history export for NORAD_CAT_ID 25544) when
# prompted.
# ==========================================================

# !pip install requests pandas numpy scikit-learn tensorflow

import requests
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import matplotlib.pyplot as plt
from google.colab import files

# ----------------------------------------------------------
# 1. LOAD DATA
# ----------------------------------------------------------

# --- 1a. Geomagnetic Kp index (NASA DONKI), same as before ---
NASA_API_KEY = "0aTvtXUulvgeFWWgtddQbBqeoepSSgZhTnd0Xia9"  # replace with your key if this expires
url = f"https://api.nasa.gov/DONKI/GST?startDate=2023-01-01&endDate=2023-12-31&api_key={NASA_API_KEY}"
response = requests.get(url)

records = []
if response.status_code == 200:
    for storm in response.json():
        for kp_entry in storm.get('allKpIndex', []):
            records.append({'Date': kp_entry['observedTime'], 'Kp_Index': kp_entry['kpIndex']})

df_kp = pd.DataFrame(records)
df_kp['Date'] = pd.to_datetime(df_kp['Date']).dt.tz_localize(None)
df_kp.set_index('Date', inplace=True)
df_kp_daily = df_kp.resample('D')['Kp_Index'].mean().ffill()

# --- 1b. ISS altitude from real Space-Track TLE history ---
print("Upload iss_25544_2023.csv (Space-Track gp_history export for NORAD 25544):")
uploaded = files.upload()
tle_filename = list(uploaded.keys())[0]

df_tle = pd.read_csv(tle_filename)
df_tle['EPOCH'] = pd.to_datetime(df_tle['EPOCH'])
df_tle['Mean_Altitude_km'] = (df_tle['PERIAPSIS'] + df_tle['APOAPSIS']) / 2
df_tle = df_tle.set_index('EPOCH').sort_index()
df_alt_daily = df_tle['Mean_Altitude_km'].resample('D').mean().ffill()

# ----------------------------------------------------------
# 2. MERGE, DETECT REBOOSTS, COMPUTE DELTA TARGET
# ----------------------------------------------------------

df_master = pd.DataFrame({'Kp_Index': df_kp_daily, 'Altitude_km': df_alt_daily}).dropna()

# Day-over-day change. delta[t] = altitude[t] - altitude[t-1]
df_master['Altitude_Delta_km'] = df_master['Altitude_km'].diff()

# Reboost = large positive jump (same threshold logic as your masking step)
REBOOST_THRESHOLD_KM = 0.5
df_master['Is_Reboost_Maneuver'] = df_master['Altitude_Delta_km'] > REBOOST_THRESHOLD_KM

df_master = df_master.dropna()  # drops the first row (no prior delta)

print(f"Merged daily samples: {len(df_master)}")
print(f"Reboost days detected: {df_master['Is_Reboost_Maneuver'].sum()}")

# ----------------------------------------------------------
# 3. SCALE INPUT (Kp) AND TARGET (delta) SEPARATELY
# ----------------------------------------------------------

kp_scaler = MinMaxScaler()
kp_scaled = kp_scaler.fit_transform(df_master[['Kp_Index']])

# StandardScaler for delta: deltas are small and centered near
# zero (mostly slow decay, occasional large positive reboost
# jumps), so zero-mean/unit-variance scaling is more appropriate
# than min-max here.
delta_scaler = StandardScaler()
delta_scaled = delta_scaler.fit_transform(df_master[['Altitude_Delta_km']])

# ----------------------------------------------------------
# 4. WINDOWING (same 10-day window as before)
# ----------------------------------------------------------

window_size = 10
X, y = [], []
for i in range(window_size, len(df_master)):
    X.append(kp_scaled[i - window_size:i, 0])   # 10 days of Kp
    y.append(delta_scaled[i, 0])                # next-day delta (scaled)

X, y = np.array(X), np.array(y)
X = np.reshape(X, (X.shape[0], X.shape[1], 1))

dates = df_master.index[window_size:]
actual_altitude = df_master['Altitude_km'].values[window_size:]
prev_altitude = df_master['Altitude_km'].values[window_size - 1:-1]  # true altitude at t-1
is_reboost = df_master['Is_Reboost_Maneuver'].values[window_size:]

split_index = int(0.8 * len(X))
X_train, X_test = X[:split_index], X[split_index:]
y_train, y_test = y[:split_index], y[split_index:]

dates_test = dates[split_index:]
actual_altitude_test = actual_altitude[split_index:]
prev_altitude_test = prev_altitude[split_index:]
is_reboost_test = is_reboost[split_index:]

# ----------------------------------------------------------
# 5. LSTM (same architecture/hyperparameters as before,
#    only the target has changed)
# ----------------------------------------------------------

model = Sequential([
    LSTM(50, return_sequences=True, input_shape=(X_train.shape[1], 1)),
    Dropout(0.2),
    LSTM(50),
    Dense(1)
])

model.compile(optimizer='adam', loss='mse')
print("Training delta-prediction LSTM...")
history = model.fit(X_train, y_train, epochs=20, batch_size=16, validation_split=0.1, verbose=1)

# ----------------------------------------------------------
# 6. PREDICT DELTA, RECONSTRUCT ABSOLUTE ALTITUDE
# ----------------------------------------------------------

pred_delta_scaled = model.predict(X_test)
pred_delta_km = delta_scaler.inverse_transform(pred_delta_scaled).flatten()

# Anchor to the TRUE previous value, same as persistence does
pred_altitude = prev_altitude_test + pred_delta_km
persistence_altitude = prev_altitude_test  # tomorrow = today

# ----------------------------------------------------------
# 7. METRICS: LSTM vs PERSISTENCE, full and reboost-excluded
# ----------------------------------------------------------

def rmse(a, b): return np.sqrt(np.mean((a - b) ** 2))
def mae(a, b): return np.mean(np.abs(a - b))

mask = ~is_reboost_test

print("\n=== FULL TEST PERIOD (n=%d) ===" % len(actual_altitude_test))
print(f"LSTM (delta)  RMSE={rmse(actual_altitude_test, pred_altitude):.4f} km   MAE={mae(actual_altitude_test, pred_altitude):.4f} km")
print(f"Persistence   RMSE={rmse(actual_altitude_test, persistence_altitude):.4f} km   MAE={mae(actual_altitude_test, persistence_altitude):.4f} km")

print("\n=== REBOOST DAYS EXCLUDED (n=%d) ===" % mask.sum())
print(f"LSTM (delta)  RMSE={rmse(actual_altitude_test[mask], pred_altitude[mask]):.4f} km   MAE={mae(actual_altitude_test[mask], pred_altitude[mask]):.4f} km")
print(f"Persistence   RMSE={rmse(actual_altitude_test[mask], persistence_altitude[mask]):.4f} km   MAE={mae(actual_altitude_test[mask], persistence_altitude[mask]):.4f} km")

# ----------------------------------------------------------
# 8. PLOT + SAVE RESULTS
# ----------------------------------------------------------

plt.figure(figsize=(12, 6))
plt.plot(dates_test, actual_altitude_test, color='blue', label='Actual Orbital Altitude')
plt.plot(dates_test, pred_altitude, color='red', linestyle='dashed', label='AI Prediction (LSTM, delta-based)')
plt.plot(dates_test, persistence_altitude, color='gray', linestyle=':', label='Persistence Baseline')
reboost_dates = dates_test[is_reboost_test]
reboost_vals = actual_altitude_test[is_reboost_test]
plt.scatter(reboost_dates, reboost_vals, color='orange', marker='*', s=200, zorder=5, label='Reboost Maneuver Detected')
plt.title('Prediction of LEO Satellite Orbital Decay: Delta-Based LSTM vs Persistence')
plt.xlabel('Date')
plt.ylabel('Mean Altitude (km)')
plt.legend()
plt.grid(True)
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('Figure_Delta_LSTM_vs_Persistence.pdf', format='pdf', bbox_inches='tight')
plt.savefig('Figure_Delta_LSTM_vs_Persistence.png', dpi=300, bbox_inches='tight')
plt.show()

results_df = pd.DataFrame({
    'Date': dates_test,
    'Actual_Altitude_km': actual_altitude_test,
    'Predicted_Altitude_km': pred_altitude,
    'Persistence_Baseline_km': persistence_altitude,
    'Predicted_Delta_km': pred_delta_km,
    'Actual_Delta_km': actual_altitude_test - prev_altitude_test,
    'Is_Reboost_Maneuver': is_reboost_test
})
results_df.to_csv('LSTM_Delta_Results.csv', index=False)

files.download('Figure_Delta_LSTM_vs_Persistence.pdf')
files.download('Figure_Delta_LSTM_vs_Persistence.png')
files.download('LSTM_Delta_Results.csv')
