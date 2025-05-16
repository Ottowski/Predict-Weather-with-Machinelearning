import pandas as pd

def data_processor(WH):
    WH = WH.rename(columns={'Formatted Date': 'date'})
    WH['date'] = pd.to_datetime(WH['date'], utc=True)
    WH['month'] = WH['date'].dt.month
    WH['weekday'] = WH['date'].dt.weekday
    WH['day'] = WH['date'].dt.day

    # Changes name for given columns
    rename_map = {
        "Temperature (C)": "avg_temp",
        "Humidity": "humidity",
        "Wind Speed (km/h)": "wind_speed",
        "Pressure (millibars)": "pressure"
    }
    WH = WH.rename(columns=rename_map)

    # remove values after columns changed  
    WH = WH.dropna(subset=["avg_temp", "humidity", "wind_speed", "pressure"])

    print("Columns after processing:", WH.columns.tolist())  

    return WH

if __name__ == "__main__":
    df = pd.read_csv("data/weatherHistory.csv")
    df_clean = data_processor(df)
    print(df_clean.head())
