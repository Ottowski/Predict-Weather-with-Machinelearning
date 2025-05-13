import argparse
import joblib
import os
import pandas as pd


# Load model
model_path = os.path.join("model", "linear_model.pkl")
model = joblib.load(model_path)

# Read argument
parser = argparse.ArgumentParser(description="Förutsäg temperatur baserat på väderdata")
parser.add_argument("--humidity", type=float, required=True, help="Luftfuktighet (t.ex. 0.75)")
parser.add_argument("--wind_speed", type=float, required=True, help="Vindhastighet i km/h (t.ex. 10.5)")
parser.add_argument("--pressure", type=float, required=True, help="Lufttryck i millibar (t.ex. 1012)")
args = parser.parse_args()

# Create Datapoint
input_data = pd.DataFrame([{
    "Humidity": args.humidity,
    "Wind Speed (km/h)": args.wind_speed,
    "Pressure (millibars)": args.pressure
}])
# Prediction
predicted_temp = model.predict(input_data)[0]
print(f"🌡️ Förutsagd temperatur: {predicted_temp:.2f} °C")
