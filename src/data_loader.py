import pandas as pd

def load_weather_data(filepath):
    try:
        WH = pd.read_csv(filepath)
        print("Good! The File is read.")
        return WH
    except FileNotFoundError:
        print(f"Oh no! File {filepath} not found.")
        return pd.DataFrame()
