import pandas as pd

def load_weather_data(filepath):
    try:
        WH = pd.read_csv(filepath)
        print("✅ File read.")
        return WH
    except FileNotFoundError:
        print(f"❌ File {filepath} not found.")
        return pd.DataFrame()
