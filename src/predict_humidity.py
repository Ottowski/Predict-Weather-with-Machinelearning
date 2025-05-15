import argparse
import joblib
import os
import pandas as pd

# Load model
model_path = os.path.join("model", "humidity_model.pkl")
model = joblib.load(model_path)

# Read arguments
parser = argparse.ArgumentParser(description="Predict humidity based on temperature, wind speed and pressure")
parser.add_argument("--temperature", type=float, required=True, help="Temperature in Celsius (e.g. 22.5)")
parser.add_argument("--wind_speed", type=float, required=True, help="Wind speed in km/h (e.g. 10.5)")
parser.add_argument("--pressure", type=float, required=True, help="Air pressure in millibar (e.g. 1012)")
args = parser.parse_args()

# Create datapoint
input_data = pd.DataFrame([{
    "Temperature (C)": args.temperature,
    "Wind Speed (km/h)": args.wind_speed,
    "Pressure (millibars)": args.pressure
}])

# Predict
predicted_humidity = model.predict(input_data)[0]
print(f"Predicted Humidity: {predicted_humidity:.3f}")