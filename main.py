from src.data_loader import load_weather_data
from src.data_model import train_linear_regression
from src.data_processor import data_processor
from src.random_forest_model import train_random_forest
from sklearn.model_selection import train_test_split
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import joblib
import os


# See if models successfully loaded 
def load_models():
    try:
        humidity_model = joblib.load("model/humidity_model.pkl")
        wind_model = joblib.load("model/wind_model.pkl")
        pressure_model = joblib.load("model/pressure_model.pkl")
        temperature_model = joblib.load("model/temperature_model.pkl")
        print("Models loaded successfully.")
        return humidity_model, wind_model, pressure_model, temperature_model
    except FileNotFoundError as e:
        print(f"Model file missing: {e}")
        return None, None, None, None


def predict_weather(temperature=None, humidity=None, wind_speed=None, pressure=None,
                    humidity_model=None, wind_model=None, pressure_model=None, temperature_model=None):
    result = {}

    if not all([humidity_model, wind_model, pressure_model, temperature_model]):
        print("One or more models not loaded. Prediction skipped.")
        return result

    # See if humidity is missing
    if humidity is None and temperature is not None and wind_speed is not None and pressure is not None:
        humidity = humidity_model.predict([[temperature, wind_speed, pressure]])[0]
        result["predicted_humidity"] = round(float(humidity), 3)

    # See if wind speed is missing
    if wind_speed is None and temperature is not None and humidity is not None and pressure is not None:
        wind_speed = wind_model.predict([[temperature, humidity, pressure]])[0]
        result["predicted_wind_speed"] = round(float(wind_speed), 2)

    # See if pressure is missing
    if pressure is None and temperature is not None and humidity is not None and wind_speed is not None:
        pressure = pressure_model.predict([[temperature, humidity, wind_speed]])[0]
        result["predicted_pressure"] = round(float(pressure), 2)

    # See if temperature is missing
    if temperature is None and humidity is not None and wind_speed is not None and pressure is not None:
        temperature = temperature_model.predict([[humidity, wind_speed, pressure]])[0]
        result["predicted_temperature"] = round(float(temperature), 2)

    # Add given values
    if temperature is not None:
        result["temperature"] = round(float(temperature), 2)
    if humidity is not None:
        result["humidity"] = round(float(humidity), 3)
    if wind_speed is not None:
        result["wind_speed"] = round(float(wind_speed), 2)
    if pressure is not None:
        result["pressure"] = round(float(pressure), 2)

    return result


def main():
    # Reads data
    WH = load_weather_data("data/weatherHistory.csv")
    print("Original Columns:")
    print(WH.columns)

    # Data Cleaning
    WH_clean = data_processor(WH)
    print("Cleaned data:")
    print(WH_clean.head())

    # Control if 'avg_temp' already exist
    if "avg_temp" not in WH_clean.columns:
        print("Column 'avg_temp' doesn't exist in the cleaned data.")
        return

    # Create features and target
    X = WH_clean.drop(["avg_temp", "date", "Summary", "Precip Type", "Daily Summary"], axis=1)
    y = WH_clean["avg_temp"]

    # Split up trained and test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Trains datamodel
    model = train_linear_regression(WH_clean)
    print("\nModel is trained and ready for use!")

    # Train Random Forest
    rf_model = train_random_forest(X_train, X_test, y_train, y_test)

    # Filter data to realistic value ranges
    WH_filtered = WH[
        (WH['Temperature (C)'] > -40) & (WH['Temperature (C)'] < 60) &
        (WH['Pressure (millibars)'] > 970) & (WH['Pressure (millibars)'] < 1050) &
        (WH['Humidity'] >= -2.0) & (WH['Humidity'] <= 4.0) &
        (WH['Wind Speed (km/h)'] >= 0) & (WH['Wind Speed (km/h)'] <= 100)
    ]

    # Check if 'Temperature (C)' exists in the data and plot pairplot and then visualize
    if 'Temperature (C)' in WH.columns:
        # Create pairplot to better visualize features
        sns.pairplot(WH_filtered[['Temperature (C)', 'Humidity', 'Wind Speed (km/h)', 'Pressure (millibars)']])
        plt.suptitle("Filtered Feature Relationships (Expanded Ranges)", y=1.02)
        plt.show()
    else:
        print("Column 'Temperature (C)' doesn't exist in the cleaned data.")

    # Load models and start prediction
    humidity_model, wind_model, pressure_model, temperature_model = load_models()
    prediction = predict_weather(
        temperature=22.5,
        wind_speed=10.5,
        pressure=1012,
        humidity_model=humidity_model,
        wind_model=wind_model,
        pressure_model=pressure_model,
        temperature_model=temperature_model
    )
    print("Prediction Result:")
    print(prediction)


if __name__ == "__main__":
    main()
