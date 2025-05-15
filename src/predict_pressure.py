import argparse
import joblib
import os
import pandas as pd

# Load model
model_path = os.path.join("model", "pressure_model.pkl")
model = joblib.load(model_path)

# Read arguments
parser = argparse.ArgumentParser(description="Predict pressure based on temperature, humidity and wind speed")
parser.add_argument("--temperature", type=float, required=True, help="Temperature in Celsius (e.g. 22.5)")
parser.add_argument("--humidity", type=float, required=True, help="Humidity (e.g. 0.75)")
parser.add_argument("--wind_speed", type=float, required=True, help="Wind speed in km/h (e.g. 10.5)")
args = parser.parse_args()

# Create datapoint
input_data = pd.DataFrame([{
    "Temperature (C)": args.temperature,
    "Humidity": args.humidity,
    "Wind Speed (km/h)": args.wind_speed
}])

# Predict
predicted_pressure = model.predict(input_data)[0]
print(f"Predicted Pressure: {predicted_pressure:.2f} millibars")