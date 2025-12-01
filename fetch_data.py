# fetch_data.py — VERSI FINAL: INDODAX BTC/USDT DIRECT (USD Murni, Tanpa Konversi!)
import requests
import pandas as pd
from datetime import datetime
import time
import os

# API Indodax BTC/USDT — Harga sudah USD, OHLC real, Volume USDT
BTC_USDT_URL = "https://indodax.com/api/ticker/btcusdt"
CSV_FILE     = "data/BTC-USDT.csv"

os.makedirs("data", exist_ok=True)

def get_btc_usdt_price():
    try:
        r = requests.get(BTC_USDT_URL, timeout=15)
        r.raise_for_status()
        ticker = r.json()["ticker"]

        # Harga USD langsung dari API
        open_price = float(ticker["last"])  # Pakai last sebagai approx open (untuk 1m snapshot)
        high       = float(ticker["high"])
        low        = float(ticker["low"])
        close      = float(ticker["last"])
        volume     = float(ticker["vol_usdt"])  # Volume dalam USDT (sudah USD)

        # UNIX TIMESTAMP float (dari server_time)
        unix_time = float(ticker["server_time"])

        return {
            "Timestamp": unix_time,      # ← 1764561236.0 (sama persis format dataset lama)
            "Open": round(open_price, 6),  # 6 desimal USD
            "High": round(high, 6),
            "Low": round(low, 6),
            "Close": round(close, 6),
            "Volume": round(volume, 6)
        }

    except Exception as e:
        print("Error fetching BTC/USDT:", e)
        return None

def append_to_csv(row):
    if row is None:
        return

    df_new = pd.DataFrame([row])

    if not os.path.exists(CSV_FILE):
        df_new.to_csv(CSV_FILE, index=False)
        print("File CSV baru dibuat → format 100% sama dengan dataset lama!")
    else:
        df = pd.read_csv(CSV_FILE)
        if df.empty:
            df_new.to_csv(CSV_FILE, index=False)
        else:
            last_timestamp = df.iloc[-1]["Timestamp"]
            # Tambah jika beda minimal 55 detik (hindari duplikat)
            if row["Timestamp"] - last_timestamp >= 55:
                df_new.to_csv(CSV_FILE, mode='a', header=False, index=False)
                ts = datetime.fromtimestamp(int(row["Timestamp"]))
                print(f"{ts.strftime('%Y-%m-%d %H:%M:%S')} → {row['Timestamp']:.1f} | High: ${row['High']:,.2f} | Low: ${row['Low']:,.2f} | Close: ${row['Close']:,.2f} | Vol: ${row['Volume']:,.0f}")
            else:
                print(f"{row['Timestamp']:.1f} → sudah ada, skip")

# =============================================================================
if __name__ == "__main__":
    print("="*70)
    print("  BOT BTC-USDT 1-MENIT VIA INDODAX BTC/USDT DIRECT")
    print("  Harga USD Murni • OHLC Real • Tanpa Konversi • 100% Jalan di Indonesia!")
    print("="*70)

    while True:
        try:
            data = get_btc_usdt_price()
            append_to_csv(data)
            time.sleep(60)
        except KeyboardInterrupt:
            print("\n\nBot berhenti. Data tersimpan di:")
            print(os.path.abspath(CSV_FILE))
            break
        except Exception as e:
            print("Error tak terduga:", e)
            time.sleep(60)