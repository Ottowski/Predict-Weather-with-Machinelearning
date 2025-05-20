# train_models.py
import pandas as pd
from sklearn.linear_model import LinearRegression
import joblib
import os

os.makedirs("predict-models", exist_ok=True)

# Read data
df = pd.read_csv(r"data/weatherHistory.csv")

# Train temperature-model
X_temp = df[["Humidity", "Wind Speed (km/h)", "Pressure (millibars)"]]
y_temp = df["Temperature (C)"]
model_temp = LinearRegression().fit(X_temp, y_temp)
joblib.dump(model_temp, "predict-models/temperature_model.pkl")

# Train humidity model
X_humidity = df[["Temperature (C)", "Wind Speed (km/h)", "Pressure (millibars)"]]
y_humidity = df["Humidity"]
model_humidity = LinearRegression().fit(X_humidity, y_humidity)
joblib.dump(model_humidity, "predict-models/humidity_model.pkl")

# Train wind speed-model
X_wind = df[["Temperature (C)", "Humidity", "Pressure (millibars)"]]
y_wind = df["Wind Speed (km/h)"]
model_wind = LinearRegression().fit(X_wind, y_wind)
joblib.dump(model_wind, "predict-models/wind_speed_model.pkl")

# Train pressure-model
X_pressure = df[["Temperature (C)", "Humidity", "Wind Speed (km/h)"]]
y_pressure = df["Pressure (millibars)"]
model_pressure = LinearRegression().fit(X_pressure, y_pressure)
joblib.dump(model_pressure, "predict-models/pressure_model.pkl")

print("All models are trained and saved in the 'predict-models/'folder.")
