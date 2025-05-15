import argparse
import joblib
import os
import pandas as pd

# Load model
model_path = os.path.join("model", "wind_speed_model.pkl")
model = joblib.load(model_path)

# Read arguments
parser = argparse.ArgumentParser(description="Predict wind speed based on temperature, humidity and pressure")
parser.add_argument("--temperature", type=float, required=True, help="Temperature in Celsius (e.g. 22.5)")
parser.add_argument("--humidity", type=float, required=True, help="Humidity (e.g. 0.75)")
parser.add_argument("--pressure", type=float, required=True, help="Air pressure in millibar (e.g. 1012)")
args = parser.parse_args()

# Create datapoint
input_data = pd.DataFrame([{
    "Temperature (C)": args.temperature,
    "Humidity": args.humidity,
    "Pressure (millibars)": args.pressure
}])

# Predict
predicted_wind_speed = model.predict(input_data)[0]
print(f"Predicted Wind Speed: {predicted_wind_speed:.2f} km/h")