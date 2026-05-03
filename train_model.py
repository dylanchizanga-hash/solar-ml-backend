import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ===============================
# LOAD DATA FROM CSV
# ===============================
df = pd.read_csv("data/solar_data_15days_10min.csv")

print("CSV loaded successfully")
print("Number of rows:", len(df))
print(df.head())

# ===============================
# CLEAN / PREPARE DATA
# ===============================
df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
df = df.dropna(subset=["timestamp"])

# Create time-based features
df["hour"] = df["timestamp"].dt.hour
df["minute"] = df["timestamp"].dt.minute
df["day"] = df["timestamp"].dt.day
df["month"] = df["timestamp"].dt.month

# Keep only numeric rows where needed
numeric_cols = [
    "irradiance",
    "temperature",
    "humidity",
    "voltage",
    "current",
    "power",
    "battery"
]

for col in numeric_cols:
    df[col] = pd.to_numeric(df[col], errors="coerce")

df = df.dropna()

# ===============================
# FEATURES AND TARGET
# ===============================
X = df[
    [
        "irradiance",
        "temperature",
        "humidity",
        "voltage",
        "current",
        "battery",
        "hour",
        "minute",
        "day",
        "month"
    ]
]

y = df["power"]

# ===============================
# TRAIN / TEST SPLIT
# ===============================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
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
model.fit(X_train, y_train)

# ===============================
# PREDICT
# ===============================
predictions = model.predict(X_test)

# ===============================
# EVALUATE
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
print("\nModel saved as model.pkl")