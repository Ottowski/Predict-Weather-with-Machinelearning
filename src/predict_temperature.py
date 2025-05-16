import argparse
import joblib
import pandas as pd

# Read arguments
parser = argparse.ArgumentParser()
parser.add_argument("--humidity", type=float, required=True)
parser.add_argument("--wind_speed", type=float, required=True)
parser.add_argument("--pressure", type=float, required=True)
args = parser.parse_args()

# Create datapoint
input_data = pd.DataFrame([{
    "Humidity": args.humidity,
    "Wind Speed (km/h)": args.wind_speed,
    "Pressure (millibars)": args.pressure
}])

# Load model and predict
model = joblib.load("models/temperature_model.pkl")
predicted_temp = model.predict(input_data)[0]
print(f"Predicted temperatur: {predicted_temp:.2f} °C")
