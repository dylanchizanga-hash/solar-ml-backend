import sys
import json
import pandas as pd
import joblib

MODEL_FILE = "model.pkl"

try:
    raw_input = sys.stdin.read()
    latest = json.loads(raw_input)

    model = joblib.load(MODEL_FILE)

    timestamp = latest.get("timestamp", "")
    parsed_timestamp = pd.to_datetime(timestamp, errors="coerce")

    if pd.isna(parsed_timestamp):
        print(json.dumps({
            "success": False,
            "error": "Invalid timestamp"
        }))
        raise SystemExit

    features = pd.DataFrame([{
        "irradiance": float(latest.get("irradiance", 0) or 0),
        "temperature": float(latest.get("temperature", 0) or 0),
        "humidity": float(latest.get("humidity", 0) or 0),
        "voltage": float(latest.get("voltage", 0) or 0),
        "current": float(latest.get("current", 0) or 0),
        "battery": float(latest.get("battery", 0) or 0),
        "hour": int(parsed_timestamp.hour),
        "minute": int(parsed_timestamp.minute),
        "day": int(parsed_timestamp.day),
        "month": int(parsed_timestamp.month),
    }])

    predicted_power = float(model.predict(features)[0])

    if predicted_power < 0:
        predicted_power = 0.0

    result = {
        "success": True,
        "timestamp": timestamp,
        "irradiance": round(float(latest.get("irradiance", 0) or 0), 3),
        "temperature": round(float(latest.get("temperature", 0) or 0), 3),
        "humidity": round(float(latest.get("humidity", 0) or 0), 3),
        "voltage": round(float(latest.get("voltage", 0) or 0), 3),
        "current": round(float(latest.get("current", 0) or 0), 3),
        "battery": round(float(latest.get("battery", 0) or 0), 3),
        "predicted_power": round(predicted_power, 3),
    }

    print(json.dumps(result))

except Exception as e:
    print(json.dumps({
        "success": False,
        "error": str(e)
    }))