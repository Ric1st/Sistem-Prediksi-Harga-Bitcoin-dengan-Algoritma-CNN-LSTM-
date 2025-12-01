# app.py - Final Version (Bersih, Stabil, Future 10 Candle Sesuai Timeframe)

from flask import Flask, render_template, jsonify
import pandas as pd
import numpy as np
import pickle
from datetime import datetime, timedelta
from tensorflow.keras.models import load_model
import warnings

# Hilangkan warning yang mengganggu
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

app = Flask(__name__)

# Load model & scaler sekali saat app start
MODEL = load_model("model/btc_usdt-2.keras")
with open("model/scaler-2.pkl", "rb") as f:
    SCALER = pickle.load(f)

WINDOW = 60
CSV = "data/BTC-USDT.csv"

# Mapping timeframe → (pandas resample rule, delta waktu untuk future)
TIMEFRAME_MAP = {
    "1m":  ("1min",  timedelta(minutes=1)),
    "5m":  ("5min",  timedelta(minutes=5)),
    "15m": ("15min", timedelta(minutes=15)),
    "1h":  ("1h",    timedelta(hours=1)),
    "4h":  ("4h",    timedelta(hours=4)),
    "1d":  ("1D",    timedelta(days=1)),
    "1w":  ("1W",    timedelta(weeks=1)),
    "1M":  ("1ME",   timedelta(days=30)),   # month-end
    "1y":  ("1YE",   timedelta(days=365)),  # approx year
}

def predict_next_steps(steps: int = 10):
    """
    Prediksi `steps` candle ke depan (autoregressive).
    Model hanya memprediksi Close, sisanya dummy 0.
    """
    df = pd.read_csv(CSV).tail(WINDOW)
    values = df[["Open", "High","Low","Close","Volume"]].values.astype("float32")
    scaled = SCALER.transform(values)
    X = scaled.reshape(1, WINDOW, 5)

    predictions = []
    current = X.copy()

    for _ in range(steps):
        pred_scaled = MODEL.predict(current, verbose=0)[0][0]
        predictions.append(pred_scaled)

        # Geser window
        shifted = current[0, 1:, :].copy()

        # Dummy row: hanya Close yang diprediksi, sisanya 0
        dummy = np.zeros((1, 5))
        dummy[0, 3] = pred_scaled
        new_scaled = SCALER.transform(dummy)

        new_window = np.vstack([shifted, new_scaled[0]])
        current = new_window.reshape(1, WINDOW, 5)

    # Inverse transform hanya kolom Close
    dummy_arr = np.zeros((len(predictions), 5))
    dummy_arr[:, 3] = predictions
    real_prices = SCALER.inverse_transform(dummy_arr)[:, 3]
    return real_prices.tolist()

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/live")
def live():
    df = pd.read_csv(CSV).tail(1)
    current_price = float(df["Close"].iloc[-1])

    # Prediksi 1 menit ke depan
    pred_1min = predict_next_steps(steps=1)[0]

    change = pred_1min - current_price
    pct = change / current_price * 100

    signal = "BUY" if change > 120 else "SELL" if change < -120 else "HOLD"
    color = "green" if signal == "BUY" else "red" if signal == "SELL" else "gray"

    return jsonify({
        "current": round(current_price, 2),
        "predicted": round(pred_1min, 2),
        "change": round(change, 2),
        "pct": round(pct, 3),
        "signal": signal,
        "color": color,
        "time": datetime.now().strftime("%H:%M:%S")
    })

@app.route("/chart_data/<tf>")
def chart_data(tf):
    df = pd.read_csv(CSV)
    df["time"] = pd.to_datetime(df["Timestamp"], unit="s")

    # Default 1h kalau tf tidak ada di map
    rule, delta = TIMEFRAME_MAP.get(tf, ("1h", timedelta(hours=1)))

    # Resample OHLCV
    ohlc = df.set_index("time").resample(rule).agg({
        "Open":  "first",
        "High":  "max",
        "Low":   "min",
        "Close": "last",
        "Volume":"sum"
    }).dropna()

    chart_data = ohlc.tail(100).reset_index()

    # Format waktu sesuai timeframe
    if tf in ["1m", "5m", "15m"]:
        chart_data["time_label"] = chart_data["time"].dt.strftime("%H:%M %d/%m")
    elif tf in ["1h", "4h"]:
        chart_data["time_label"] = chart_data["time"].dt.strftime("%H:%M %d/%m")
    else:
        chart_data["time_label"] = chart_data["time"].dt.strftime("%d/%m/%Y")

    # Prediksi 10 candle ke depan
    future_prices = predict_next_steps(steps=10)

    # Waktu terakhir di dataset
    last_ts = df["Timestamp"].iloc[-1]
    last_time = datetime.fromtimestamp(last_ts)

    future = []
    t = last_time
    for i in range(10):
        t += delta
        if tf in ["1m", "5m", "15m"]:
            label = t.strftime("%H:%M")
        elif tf in ["1h", "4h"]:
            label = t.strftime("%H:%M %d/%m")
        else:
            label = t.strftime("%d/%m/%Y")
        future.append({
            "time": label,
            "price": round(future_prices[i], 2)
        })

    return jsonify({
        "actual": [
            {
                "time":  row["time_label"],
                "open":  round(row["Open"], 2),
                "high":  round(row["High"], 2),
                "low":   round(row["Low"], 2),
                "close": round(row["Close"], 2)
            } for _, row in chart_data.iterrows()
        ],
        "future": future
    })

if __name__ == "__main__":
    app.run(debug=False, port=5000)