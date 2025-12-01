# File ini akan melakukan pembandingan kinerja antara tiga arsitektur deep learning yang umum digunakan untuk data deret waktu: LSTM (Long Short-Term Memory), GRU (Gated Recurrent Unit), dan CNN-1D (Convolutional Neural Network 1D). Outputnya akan menunjukkan model mana yang memiliki nilai error (RMSE dan MAE) terendah.

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, GRU, Dense, Dropout, Conv1D, MaxPooling1D, Flatten
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import mean_squared_error, mean_absolute_error
import time
import os
import sys
import matplotlib.pyplot as plt # PENTING: Import untuk visualisasi

# Atur logging TensorFlow agar tidak terlalu verbose saat training
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# --- Konfigurasi ---
DATA_FILE = 'BTC-USDT.csv' # Menggunakan nama file yang diindikasikan: 'BTC-USDT.csv'
TIME_STEP = 60 # Jumlah langkah waktu (60 data terakhir) sebagai input
PREDICT_COLUMN = 'Close' # Kolom target prediksi (menggunakan 'Close')
EPOCHS = 5 # Gunakan epoch yang kecil untuk perbandingan
BATCH_SIZE = 128
PATIENCE = 5 # Early Stopping
MAX_RECORDS = 100000 # BATAS BARU: Hanya proses 100.000 data terakhir untuk perbandingan cepat

# --- 1. Fungsi Persiapan Data dan Sekuens ---

def create_sequences(data, time_step, target_col_index):
    """
    Mengubah data deret waktu yang dinormalisasi menjadi pasangan input (X) 
    dan output (y) untuk model.
    """
    X, y = [], []
    for i in range(len(data) - time_step):
        X.append(data[i:(i + time_step), :])
        y.append(data[i + time_step, target_col_index])
    return np.array(X), np.array(y)

# --- 2. Definisi Model ---

def build_lstm_model(input_shape):
    """Membangun model Long Short-Term Memory (LSTM)."""
    model = Sequential([
        LSTM(units=50, return_sequences=True, input_shape=input_shape),
        Dropout(0.2),
        LSTM(units=50, return_sequences=False),
        Dropout(0.2),
        Dense(units=1) 
    ])
    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])
    return model

def build_gru_model(input_shape):
    """Membangun model Gated Recurrent Unit (GRU)."""
    model = Sequential([
        GRU(units=50, return_sequences=True, input_shape=input_shape),
        Dropout(0.2),
        GRU(units=50, return_sequences=False),
        Dropout(0.2),
        Dense(units=1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])
    return model

def build_cnn_1d_model(input_shape):
    """Membangun model Convolutional Neural Network 1D."""
    model = Sequential([
        Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=input_shape), 
        MaxPooling1D(pool_size=2),
        Flatten(), 
        Dense(50, activation='relu'),
        Dropout(0.2),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])
    return model

# --- 3. Fungsi Pelatihan dan Evaluasi ---

def train_and_evaluate(model_name, model, X_train, y_train, X_test, y_test, scaler, target_col_index):
    """Melatih model, memprediksi, dan menginversi hasil ke harga asli."""
    print(f"\n--- Melatih Model: {model_name} ---")
    
    early_stopping = EarlyStopping(monitor='val_loss', patience=PATIENCE, restore_best_weights=True)
    start_time = time.time()
    
    model.fit(
        X_train, y_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_data=(X_test, y_test),
        callbacks=[early_stopping],
        verbose=1
    )
    
    training_time = time.time() - start_time
    y_pred_scaled = model.predict(X_test, verbose=0)
    
    # --- Inversi Scaling ---
    num_features = scaler.scale_.shape[0]

    # Inversi y_test (harga asli)
    y_test_dummy = np.zeros((len(y_test), num_features))
    y_test_dummy[:, target_col_index] = y_test
    y_test_inverse = scaler.inverse_transform(y_test_dummy)[:, target_col_index]

    # Inversi y_pred (harga prediksi)
    y_pred_dummy = np.zeros((len(y_pred_scaled), num_features))
    y_pred_dummy[:, target_col_index] = y_pred_scaled.flatten()
    y_pred_inverse = scaler.inverse_transform(y_pred_dummy)[:, target_col_index]
    
    # Evaluasi Metrik
    rmse = np.sqrt(mean_squared_error(y_test_inverse, y_pred_inverse))
    mae = mean_absolute_error(y_test_inverse, y_pred_inverse)
    
    print(f"\n--- Hasil {model_name} ---")
    print(f"Waktu Pelatihan: {training_time:.2f} detik")
    print(f"RMSE: ${rmse:.4f}")
    print(f"MAE: ${mae:.4f}")
    
    return {
        'model_name': model_name, 
        'rmse': rmse, 
        'mae': mae, 
        'time': training_time,
        # Mengembalikan nilai inversi untuk plotting
        'y_test_inverse': y_test_inverse, 
        'y_pred_inverse': y_pred_inverse
    }

# --- 4. Fungsi Plotting ---

def plot_comparison(model_name, y_test, y_pred):
    """Membuat plot perbandingan antara data asli dan prediksi."""
    plt.figure(figsize=(14, 7))
    plt.plot(y_test, label='Harga Asli (Test)', color='blue', linewidth=2)
    plt.plot(y_pred, label='Harga Prediksi', color='red', linestyle='--', linewidth=2)
    plt.title(f'Perbandingan Harga Asli vs Prediksi ({model_name})', fontsize=16)
    plt.xlabel('Langkah Waktu (Data Uji)', fontsize=12)
    plt.ylabel(f'Harga {PREDICT_COLUMN}', fontsize=12)
    plt.legend(fontsize=10)
    plt.grid(True, linestyle=':', alpha=0.7)
    plt.tight_layout()
    plt.show()

# --- 5. Fungsi Utama ---

def main():
    print("Memulai Perbandingan Algoritma Time Series (LSTM vs GRU vs CNN-1D)...")
    
    try:
        # Muat Data
        df = pd.read_csv(DATA_FILE)
        
        # --- Solusi untuk Data Besar: Data Sampling ---
        if len(df) > MAX_RECORDS:
            print(f"Dataset memiliki {len(df)} baris. Memproses hanya {MAX_RECORDS} baris terakhir untuk perbandingan cepat.")
            # Mengambil 100.000 data terbaru (tail)
            df = df.tail(MAX_RECORDS)
        # ---------------------------------------------

        # Data Cleaning
        df['Date'] = pd.to_datetime(df['Timestamp'], unit='s')
        df.set_index('Date', inplace=True)
        df = df.drop(columns=['Timestamp']).sort_index() 
        df.dropna(inplace=True) 
        
    except FileNotFoundError:
        print(f"ERROR: File '{DATA_FILE}' tidak ditemukan.")
        sys.exit(1)
    except Exception as e:
        print(f"ERROR saat memuat data: {e}")
        sys.exit(1)

    # Normalisasi Data
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(df)
    
    # Menemukan indeks kolom target 'Close'
    target_col_index = df.columns.get_loc(PREDICT_COLUMN)
    
    # Pembentukan Sekuens
    X, y = create_sequences(scaled_data, TIME_STEP, target_col_index)
    
    # Split Data (80% Train, 20% Test)
    split_index = int(0.8 * len(X))
    X_train, X_test = X[:split_index], X[split_index:]
    y_train, y_test = y[:split_index], y[split_index:]
    
    # Mendefinisikan Shape Input
    input_shape = (X_train.shape[1], X_train.shape[2]) 
    
    models_to_compare = [
        ('LSTM', build_lstm_model(input_shape)),
        ('GRU', build_gru_model(input_shape)),
        ('CNN-1D', build_cnn_1d_model(input_shape))
    ]
    
    results = []
    
    # Proses Pelatihan dan Evaluasi
    for name, model in models_to_compare:
        result = train_and_evaluate(name, model, X_train, y_train, X_test, y_test, scaler, target_col_index)
        results.append(result)
        
        # Panggil fungsi plot untuk visualisasi
        plot_comparison(name, result['y_test_inverse'], result['y_pred_inverse'])
    
    # Menampilkan Hasil Perbandingan
    print("\n==============================================")
    print("       RANGKUMAN HASIL PERBANDINGAN MODEL      ")
    print("==============================================")
    
    best_model_name = ""
    best_mae = float('inf') 
    
    for r in results:
        print(f"Model: {r['model_name']:<8} | RMSE: ${r['rmse']:.4f} | MAE: ${r['mae']:.4f} | Waktu: {r['time']:.2f} detik")
        if r['mae'] < best_mae:
            best_mae = r['mae']
            best_model_name = r['model_name']
            
    print("==============================================")
    print(f"REKOMENDASI: Algoritma Terbaik (berdasarkan MAE): {best_model_name}")
    print("==============================================")


if __name__ == '__main__':
    main()