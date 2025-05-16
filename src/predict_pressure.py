import argparse
import joblib
import pandas as pd

# Read arguments
parser = argparse.ArgumentParser()
parser.add_argument("--temperature", type=float, required=True)
parser.add_argument("--humidity", type=float, required=True)
parser.add_argument("--wind_speed", type=float, required=True)
args = parser.parse_args()

# Create datapoint
input_data = pd.DataFrame([{
    "Temperature (C)": args.temperature,
    "Humidity": args.humidity,
    "Wind Speed (km/h)": args.wind_speed,
}])

# Load model and predict
model = joblib.load("predict-models/pressure_model.pkl")
predicted_pressure = model.predict(input_data)[0]
print(f"Predicted Pressure: {predicted_pressure:.2f} millibars")
