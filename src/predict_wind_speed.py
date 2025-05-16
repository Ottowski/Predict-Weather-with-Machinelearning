import argparse
import joblib
import pandas as pd

# Read arguments
parser = argparse.ArgumentParser()
parser.add_argument("--temperature", type=float, required=True)
parser.add_argument("--humidity", type=float, required=True)
parser.add_argument("--pressure", type=float, required=True)
args = parser.parse_args()

# Create datapoint
input_data = pd.DataFrame([{
    "Temperature (C)": args.temperature,
    "Humidity": args.humidity,
    "Pressure (millibars)": args.pressure
}])

# Load model and predict
model = joblib.load("models/wind_speed_model.pkl")
predicted_wind_speed = model.predict(input_data)[0]
print(f"Predicted Wind Speed: {predicted_wind_speed:.2f} km/h")
