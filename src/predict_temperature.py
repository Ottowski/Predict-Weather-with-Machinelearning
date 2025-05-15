import argparse
import joblib
import os
import pandas as pd

# Load model
model_path = os.path.join("model", "linear_model.pkl")
model = joblib.load(model_path)

# Read arguments
parser = argparse.ArgumentParser(description="Predict temperature based on weather data")
parser.add_argument("--humidity", type=float, required=True, help="Humidity (e.g. 0.75)")
parser.add_argument("--wind_speed", type=float, required=True, help="Wind speed in km/h (e.g. 10.5)")
parser.add_argument("--pressure", type=float, required=True, help="Air pressure in millibar (e.g. 1012)")
args = parser.parse_args()

# Create datapoint
input_data = pd.DataFrame([{
    "Humidity": args.humidity,
    "Wind Speed (km/h)": args.wind_speed,
    "Pressure (millibars)": args.pressure
}])

# Predict
predicted_temp = model.predict(input_data)[0]
print(f"Predicted temperatur: {predicted_temp:.2f} °C")