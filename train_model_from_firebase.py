import requests
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    mean_absolute_percentage_error,
)

FIREBASE_URL = "https://solar-ml-system-default-rtdb.firebaseio.com/solar_readings.json"

print("Downloading data from Firebase...")

response = requests.get(FIREBASE_URL, timeout=30)
response.raise_for_status()
firebase_data = response.json()

if not firebase_data:
    print("No Firebase data found.")
    raise SystemExit

rows = []

for key, value in firebase_data.items():
    if not isinstance(value, dict):
        continue

    timestamp = str(value.get("timestamp", ""))

    if "T" not in timestamp:
        continue

    rows.append({
        "timestamp": timestamp,
        "irradiance": value.get("irradiance", 0),
        "temperature": value.get("temperature", 0),
        "humidity": value.get("humidity", 0),
        "voltage": value.get("voltage", 0),
        "current": value.get("current", 0),
        "power": value.get("power", 0),
        "battery": value.get("battery", 0),
    })

df = pd.DataFrame(rows)

if df.empty:
    print("No valid Firebase records found.")
    raise SystemExit

print("\nFirebase data loaded successfully")
print("Rows before cleaning:", len(df))

df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
df = df.dropna(subset=["timestamp"])

numeric_cols = [
    "irradiance",
    "temperature",
    "humidity",
    "voltage",
    "current",
    "power",
    "battery",
]

for col in numeric_cols:
    df[col] = pd.to_numeric(df[col], errors="coerce")

df = df.dropna(subset=numeric_cols)
df = df.drop_duplicates(subset=["timestamp"])

df = df[df["irradiance"] > 0]
df = df[df["voltage"] > 0]
df = df[df["current"] > 0]
df = df[df["power"] > 0]
df = df[df["battery"] > 0]
df = df[df["temperature"] > -10]
df = df[df["temperature"] < 80]
df = df[df["humidity"] >= 0]
df = df[df["humidity"] <= 100]

df = df.sort_values("timestamp")

print("Rows after cleaning:", len(df))

if len(df) < 20:
    print("Not enough valid data after cleaning.")
    raise SystemExit

df["hour"] = df["timestamp"].dt.hour
df["minute"] = df["timestamp"].dt.minute
df["day"] = df["timestamp"].dt.day
df["month"] = df["timestamp"].dt.month

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

X = df[feature_cols]
y = df["power"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
)

models = {
    "Linear Regression": LinearRegression(),
    "Random Forest Regressor": RandomForestRegressor(
        n_estimators=100,
        random_state=42,
    ),
    "Gradient Boosting Regressor": GradientBoostingRegressor(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=3,
        random_state=42,
    ),
}

results = []
best_model_name = None
best_model = None
best_r2 = -999

print("\nTraining and evaluating models...")
print("--------------------------------")

for name, model in models.items():
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    mae = mean_absolute_error(y_test, predictions)
    mse = mean_squared_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)
    mape = mean_absolute_percentage_error(y_test, predictions) * 100

    results.append({
        "Model": name,
        "MAE": mae,
        "MSE": mse,
        "R2": r2,
        "MAPE": mape,
    })

    print(f"\n{name}")
    print(f"MAE: {mae:.3f}")
    print(f"MSE: {mse:.3f}")
    print(f"R2 Score: {r2:.3f}")
    print(f"MAPE: {mape:.2f}%")

    if r2 > best_r2:
        best_r2 = r2
        best_model_name = name
        best_model = model

results_df = pd.DataFrame(results)

print("\nSummary Table")
print("-------------")
print(results_df.to_string(index=False))

joblib.dump(best_model, "model.pkl")

print(f"\nBest model saved as model.pkl")
print(f"Best model: {best_model_name}")
