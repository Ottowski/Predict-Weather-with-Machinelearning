import pandas as pd

def load_weather_data(filepath):
    # See if file is loaded or not
    try:
        WH = pd.read_csv(filepath)
        print("Good! The file is read succesfully.")
        return WH
    except FileNotFoundError:
        print(f"Oh no! File {filepath} not found.")
        return pd.DataFrame()
