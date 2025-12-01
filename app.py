# app.py — WEBSITE + PREDIKSI REAL-TIME TIAP MENIT (ALL IN ONE)
from flask import Flask, render_template, jsonify
import pandas as pd
import numpy as np
import pickle
import threading
import time
from datetime import datetime
from tensorflow.keras.models import load_model

app = Flask(__name__)

# === GANTI SESUAI MODEL KAMU ===
MODEL_PATH   = "model/btc_usdt-2.keras"     
SCALER_PATH  = "model/scaler-2.pkl"         
CSV_PATH     = "data/BTC-USDT.csv"
WINDOW       = 60

# Load model sekali di awal
print("Loading model & scaler...")
model = load_model(MODEL_PATH)
with open(SCALER_PATH, "rb") as f:
    scaler = pickle.load(f)

# Global variable untuk simpan prediksi terbaru
latest_prediction = {
    "time": "...",
    "current": 0.0,
    "predicted": 0.0,
    "change": 0.0,
    "pct": 0.0,
    "signal": "HOLD",
    "color": "gray"
}

def predict_forever():
    """Background thread: update prediksi tiap 60 detik"""
    global latest_prediction
    while True:
        try:
            df = pd.read_csv(CSV_PATH).tail(WINDOW)
            if len(df) < WINDOW:
                time.sleep(10)
                continue

            X = scaler.transform(df[["Open","High","Low","Close","Volume"]].values.astype('float32'))
            X = X.reshape(1, WINDOW, 5)

            pred_scaled = model.predict(X, verbose=0)[0][0]
            dummy = np.zeros((1,5))
            dummy[0,3] = pred_scaled
            pred_price = scaler.inverse_transform(dummy)[0,3]

            current = df.iloc[-1]["Close"]
            change = pred_price - current
            pct = change / current * 100

            signal = "STRONG BUY" if change > 120 else "BUY" if change > 0 else "STRONG SELL" if change < -120 else "SELL" if change < 0 else "HOLD"
            color = "green" if change > 0 else "red" if change < 0 else "gray"

            latest_prediction = {
                "time": datetime.now().strftime("%H:%M:%S"),
                "current": round(current, 2),
                "predicted": round(pred_price, 2),
                "change": round(change, 2),
                "pct": round(pct, 3),
                "signal": signal,
                "color": color
            }

            print(f"{latest_prediction['time']} | ${current:,.0f} → ${pred_price:,.0f} | {signal} {change:+.0f} ({pct:+.2f}%)")

        except Exception as e:
            print("Error prediksi:", e)

        time.sleep(60)  # tiap menit

# Jalankan background thread
threading.Thread(target=predict_forever, daemon=True).start()

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/live")
def live_data():
    return jsonify(latest_prediction)

@app.route("/chart_data")
def chart_data():
    try:
        df = pd.read_csv(CSV_PATH).tail(500)  # 500 menit terakhir
        data = []
        for _, row in df.iterrows():
            data.append({
                "time": datetime.fromtimestamp(row["Timestamp"]).strftime("%H:%M"),
                "price": round(row["Close"], 2)
            })
        return jsonify(data)
    except:
        return jsonify([])

if __name__ == "__main__":
    print("Website + Prediksi Live AKTIF → http://127.0.0.1:5000")
    app.run(host="0.0.0.0", port=5000, debug=False)