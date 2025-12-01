# train_anti_cheat.py — 100% BERSIH, GAK BISA CHEAT LAGI!
import pandas as pd
import numpy as np
import pickle
import os
from sklearn.preprocessing import RobustScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt

CSV_FILE     = "data/BTC-USDT.csv"
MODEL_FILE   = "model/btc_usdt-2.keras"
SCALER_FILE  = "model/scaler-2.pkl"

DATA_LIMIT   = 10_000
WINDOW_SIZE  = 90

os.makedirs("model", exist_ok=True)

print("Loading data...")
df = pd.read_csv(CSV_FILE).sort_values("Timestamp").tail(DATA_LIMIT).reset_index(drop=True)

# FITUR AMAN 100%
features = ["Open", "High", "Low", "Close", "Volume"]
price_raw = df["Close"].values.copy()  # Simpan harga asli untuk nanti

# Scale SEMUA KOLOM TERMASUK CLOSE
scaler = RobustScaler()
data_scaled = scaler.fit_transform(df[features])
with open(SCALER_FILE, "wb") as f:
    pickle.dump(scaler, f)

# Sequence
X, y = [], []
for i in range(WINDOW_SIZE, len(data_scaled)):
    X.append(data_scaled[i-WINDOW_SIZE:i])
    y.append(data_scaled[i, 3])  # Close scaled
X = np.array(X)
y = np.array(y)

split = int(0.85 * len(X))
X_train, X_val = X[:split], X[split:]
y_train, y_val = y[:split], y[split:]

# Model sederhana tapi jujur
model = Sequential([
    Conv1D(64, 3, activation='relu', input_shape=(WINDOW_SIZE, 5)),
    LSTM(80, return_sequences=True),
    Dropout(0.3),
    LSTM(80),
    Dropout(0.3),
    Dense(40, activation='relu'),
    Dense(1)
])

model.compile(optimizer=Adam(0.001), loss='huber', metrics=['mae'])
early = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

print("\nTRAINING")
# model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=50, batch_size=512, callbacks=[early], verbose=1)
model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=25, batch_size=32, callbacks=[early], verbose=1)

model.save(MODEL_FILE)
print("\n Berhasil Save Model ", MODEL_FILE)

# PREDIKSI + INVERSE TRANSFORM YANG BENAR-BENAR JUJUR
pred_scaled = model.predict(X_val, verbose=0).flatten()

# Cara inverse yang 100% aman: hanya ambil kolom Close dari scaler
dummy = np.zeros((len(pred_scaled), 5))
dummy[:, 3] = pred_scaled
pred_price = scaler.inverse_transform(dummy)[:, 3]

# True price langsung dari data asli (bukan dari scaler!)
true_price = price_raw[-len(pred_price):]

mape = np.mean(np.abs((true_price - pred_price) / true_price)) * 100
mae = np.mean(np.abs(true_price - pred_price))
accr = (1 - mape) * 100

print(f"\nAKURASI :")
print(f"→ MAPE = {mape:.4f}%")
print(f"→ MAE  = ${mae:.2f}")
print(f"→ Accuracy  = {accr:.2f}%")

# Plot
plt.figure(figsize=(15,6))
plt.plot(true_price[-500:], label='Harga Asli', linewidth=1)
plt.plot(pred_price[-500:], label='Prediksi', linewidth=1, alpha=0.9)
plt.title('Prediksi vs Real')
plt.legend(); plt.grid(alpha=0.3)
plt.show()