import os
import joblib
import pandas as pd
from src.data_loader import load_weather_data
from src.data_processor import data_processor
from sklearn.linear_model import LinearRegression


def train_and_save_models():
    # Load and clean data
    df = load_weather_data("data/weatherHistory.csv")
    df_clean = data_processor(df)

    print("Good! The File is read.")

    # control if coluns exist 
    expected_columns = ["avg_temp", "humidity", "wind_speed", "pressure"]
    for col in expected_columns:
        if col not in df_clean.columns:
            raise KeyError(f"Column '{col}' missing in cleaned data.")

    # Prepare input/output for every model 

    X_humidity = df_clean[["avg_temp", "wind_speed", "pressure"]]
    y_humidity = df_clean["humidity"]

    X_wind = df_clean[["avg_temp", "humidity", "pressure"]]
    y_wind = df_clean["wind_speed"]

    X_pressure = df_clean[["avg_temp", "humidity", "wind_speed"]]
    y_pressure = df_clean["pressure"]

    X_temp = df_clean[["humidity", "wind_speed", "pressure"]]
    y_temp = df_clean["avg_temp"]

    # Train models
    humidity_model = LinearRegression().fit(X_humidity, y_humidity)
    wind_model = LinearRegression().fit(X_wind, y_wind)
    pressure_model = LinearRegression().fit(X_pressure, y_pressure)
    temperature_model = LinearRegression().fit(X_temp, y_temp)

    # Creates folder, if already doesn't exist
    os.makedirs("model", exist_ok=True)

    # Save models
    joblib.dump(humidity_model, "model/humidity_model.pkl")
    joblib.dump(wind_model, "model/wind_model.pkl")
    joblib.dump(pressure_model, "model/pressure_model.pkl")
    joblib.dump(temperature_model, "model/temperature_model.pkl")

    print("All models have been trained and been saved in the 'model/' folder.")


if __name__ == "__main__":
    train_and_save_models()
