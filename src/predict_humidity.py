import argparse
import joblib
import pandas as pd

# Read arguments
parser = argparse.ArgumentParser()
parser.add_argument("--temperature", type=float, required=True)
parser.add_argument("--wind_speed", type=float, required=True)
parser.add_argument("--pressure", type=float, required=True)
args = parser.parse_args()

# Create datapoint
input_data = pd.DataFrame([{
    "Temperature (C)": args.temperature,
    "Wind Speed (km/h)": args.wind_speed,
    "Pressure (millibars)": args.pressure
}])

# Load model and predict
model = joblib.load("predict-models/humidity_model.pkl")
predicted_humidity = model.predict(input_data)[0]
print(f"Predicted Humidity: {predicted_humidity:.3f}")
