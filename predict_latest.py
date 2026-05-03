import json
import joblib
import pandas as pd
import requests

# ===============================
# SETTINGS
# ===============================
FIREBASE_URL = "https://solar-ml-system-default-rtdb.firebaseio.com/solar_readings.json"
MODEL_FILE = "model.pkl"
CSV_FILE = "data/solar_data_15days_10min.csv"

TARIFF_USD_PER_KWH = 0.10
INTERVAL_MINUTES = 10

# ===============================
# LOAD MODEL
# ===============================
model = joblib.load(MODEL_FILE)

# ===============================
# LOAD CSV (for daily yield)
# ===============================
df = pd.read_csv(CSV_FILE)

df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
df = df.dropna(subset=["timestamp"])

df["hour"] = df["timestamp"].dt.hour
df["minute"] = df["timestamp"].dt.minute
df["day"] = df["timestamp"].dt.day
df["month"] = df["timestamp"].dt.month
df["date_only"] = df["timestamp"].dt.date

feature_cols = [
    "irradiance",
    "temperature",
    "humidity",
    "voltage",
    "current",
    "battery",
    "hour",
    "minute",
    "day",
    "month",
]

X_hist = df[feature_cols]

# Predict historical power
df["predicted_power_w"] = model.predict(X_hist).clip(min=0)

interval_hours = INTERVAL_MINUTES / 60
df["predicted_energy_kwh"] = (df["predicted_power_w"] * interval_hours) / 1000

daily_yield = df.groupby("date_only")["predicted_energy_kwh"].sum()

average_daily_yield_kwh = float(daily_yield.mean())
predicted_30_day_revenue = average_daily_yield_kwh * 30 * TARIFF_USD_PER_KWH

# ===============================
# FETCH FIREBASE DATA
# ===============================
response = requests.get(FIREBASE_URL, timeout=20)
response.raise_for_status()

firebase_data = response.json()

rows = []

for key, value in firebase_data.items():
    if not isinstance(value, dict):
        continue

    timestamp = str(value.get("timestamp", ""))

    # 🔥 ONLY accept proper timestamps like 2026-04-29T15:25:29
    if "T" not in timestamp:
        continue

    parsed_timestamp = pd.to_datetime(timestamp, errors="coerce")

    if pd.isna(parsed_timestamp):
        continue

    rows.append({
        "timestamp": timestamp,
        "parsed_timestamp": parsed_timestamp,
        "irradiance": float(value.get("irradiance", 0) or 0),
        "temperature": float(value.get("temperature", 0) or 0),
        "humidity": float(value.get("humidity", 0) or 0),
        "voltage": float(value.get("voltage", 0) or 0),
        "current": float(value.get("current", 0) or 0),
        "battery": float(value.get("battery", 0) or 0),
    })

# ===============================
# CHECK DATA
# ===============================
if not rows:
    print(json.dumps({
        "success": False,
        "error": "No valid Firebase records found"
    }))
    exit()

# ===============================
# GET LATEST VALID RECORD
# ===============================
live_df = pd.DataFrame(rows)
live_df = live_df.sort_values("parsed_timestamp")

latest = live_df.iloc[-1]
latest_ts = latest["parsed_timestamp"]

# ===============================
# BUILD FEATURES
# ===============================
latest_features = pd.DataFrame([{
    "irradiance": latest["irradiance"],
    "temperature": latest["temperature"],
    "humidity": latest["humidity"],
    "voltage": latest["voltage"],
    "current": latest["current"],
    "battery": latest["battery"],
    "hour": int(latest_ts.hour),
    "minute": int(latest_ts.minute),
    "day": int(latest_ts.day),
    "month": int(latest_ts.month),
}])

# ===============================
# PREDICT
# ===============================
predicted_power = float(model.predict(latest_features)[0])
predicted_power = max(0, predicted_power)

predicted_hourly_energy_kwh = predicted_power / 1000

# ===============================
# FINAL OUTPUT
# ===============================
result = {
    "success": True,
    "timestamp": latest["timestamp"],
    "irradiance": round(latest["irradiance"], 3),
    "temperature": round(latest["temperature"], 3),
    "humidity": round(latest["humidity"], 3),
    "voltage": round(latest["voltage"], 3),
    "current": round(latest["current"], 3),
    "battery": round(latest["battery"], 3),
    "predicted_power": round(predicted_power, 3),
    "predicted_hourly_energy_kWh": round(predicted_hourly_energy_kwh, 4),
    "predicted_daily_yield_kWh": round(average_daily_yield_kwh, 4),
    "predicted_revenue_usd": round(predicted_30_day_revenue, 3),
}

print(json.dumps(result))