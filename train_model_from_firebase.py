import requests
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ===============================
# FIREBASE SETTINGS
# ===============================
FIREBASE_URL = "https://solar-ml-system-default-rtdb.firebaseio.com/solar_readings.json"

# ===============================
# DOWNLOAD FIREBASE DATA
# ===============================
print("Downloading data from Firebase...")

response = requests.get(FIREBASE_URL, timeout=30)
response.raise_for_status()

firebase_data = response.json()

if not firebase_data:
    print("No Firebase data found.")
    raise SystemExit

# ===============================
# CONVERT FIREBASE DATA TO ROWS
# ===============================
rows = []

for key, value in firebase_data.items():

    if not isinstance(value, dict):
        continue

    timestamp = str(value.get("timestamp", ""))

    # Ignore invalid timestamps
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

# ===============================
# CREATE DATAFRAME
# ===============================
df = pd.DataFrame(rows)

if df.empty:
    print("No valid Firebase records found.")
    raise SystemExit

print("\nFirebase data loaded successfully")
print("Rows before cleaning:", len(df))

# ===============================
# CLEAN TIMESTAMPS
# ===============================
df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
df = df.dropna(subset=["timestamp"])

# ===============================
# CONVERT TO NUMERIC
# ===============================
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

# ===============================
# REMOVE DUPLICATE TIMESTAMPS
# ===============================
df = df.drop_duplicates(subset=["timestamp"])

# ===============================
# REMOVE INVALID / ZERO DATA
# ===============================

# Keep only realistic solar generation rows
df = df[df["irradiance"] > 0]
df = df[df["voltage"] > 0]
df = df[df["current"] > 0]
df = df[df["power"] > 0]

# Environmental ranges
df = df[df["temperature"] > -10]
df = df[df["temperature"] < 80]

df = df[df["humidity"] >= 0]
df = df[df["humidity"] <= 100]

df = df[df["battery"] > 0]

# ===============================
# SORT BY TIME
# ===============================
df = df.sort_values("timestamp")

print("Rows after cleaning:", len(df))

if len(df) < 20:
    print("Not enough valid data after cleaning.")
    raise SystemExit

# ===============================
# CREATE TIME FEATURES
# ===============================
df["hour"] = df["timestamp"].dt.hour
df["minute"] = df["timestamp"].dt.minute
df["day"] = df["timestamp"].dt.day
df["month"] = df["timestamp"].dt.month

# ===============================
# FEATURES AND TARGET
# ===============================
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

# ===============================
# TRAIN TEST SPLIT
# ===============================
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# ===============================
# MODEL
# ===============================
model = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)

# ===============================
# TRAIN MODEL
# ===============================
print("\nTraining model...")

model.fit(X_train, y_train)

# ===============================
# PREDICTIONS
# ===============================
predictions = model.predict(X_test)

# ===============================
# EVALUATION
# ===============================
mae = mean_absolute_error(y_test, predictions)
mse = mean_squared_error(y_test, predictions)
r2 = r2_score(y_test, predictions)

print("\nModel Evaluation")
print("-----------------")
print(f"MAE: {mae:.3f}")
print(f"MSE: {mse:.3f}")
print(f"R2 Score: {r2:.3f}")

# ===============================
# SAVE MODEL
# ===============================
joblib.dump(model, "model.pkl")

print("\nModel trained successfully from Firebase.")
print("Cleaned dataset used.")
print("Model saved as model.pkl")