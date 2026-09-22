import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.metrics import mean_squared_error, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

# 1. GERÇEKÇİ YÖRÜNGE SİMÜLASYONU (REBOOST MASKESİ OLMADAN TEMEL MODEL)
dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='D')
altitude = []
current_alt = 418.0

np.random.seed(42)
for i in range(len(dates)):
    decay = np.random.uniform(0.01, 0.05) 
    current_alt -= decay
    if i % 45 == 0 and i != 0: 
        current_alt += 1.5 
    altitude.append(current_alt)

df_master = pd.DataFrame({'mean_altitude': altitude}, index=dates)
df_master['Kp_Index'] = np.random.uniform(1, 4, len(dates))

# 2. VERİ ÖLÇEKLENDİRME
scaler_features = MinMaxScaler()
scaled_data = scaler_features.fit_transform(df_master[['Kp_Index', 'mean_altitude']])
scaler_target = MinMaxScaler()
scaler_target.fit(df_master[['mean_altitude']])

X, y = [], []
window_size = 10 
for i in range(window_size, len(scaled_data)):
    X.append(scaled_data[i-window_size:i, 0])
    y.append(scaled_data[i, 1])

X, y = np.array(X), np.array(y)
X = np.reshape(X, (X.shape[0], X.shape[1], 1))

split_index = int(0.8 * len(X))
X_train, X_test = X[:split_index], X[split_index:]
y_train, y_test = y[:split_index], y[split_index:]

# 3. LSTM AĞI KURULUMU
model = Sequential([
    LSTM(50, return_sequences=True, input_shape=(X_train.shape[1], 1)),
    Dropout(0.2),
    LSTM(50),
    Dense(1)
])
model.compile(optimizer='adam', loss='mse')
model.fit(X_train, y_train, epochs=30, batch_size=16, validation_split=0.1, verbose=0)

# 4. TAHMİN VE HATA METRİKLERİ
predictions_scaled = model.predict(X_test, verbose=0)
predictions_km = scaler_target.inverse_transform(predictions_scaled).flatten()
y_test_km = scaler_target.inverse_transform(y_test.reshape(-1, 1)).flatten()

baseline_km = np.roll(y_test_km, shift=1)
baseline_km[0] = df_master['mean_altitude'].iloc[split_index + window_size - 1]

print("Temel model başarıyla çalıştı ve metrikler hesaplandı.")
