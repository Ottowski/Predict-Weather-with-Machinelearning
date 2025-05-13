import pandas as pd

def data_processor(WH):
    WH = WH.rename(columns={'Formatted Date': 'date'})
    WH['date'] = pd.to_datetime(WH['date'], utc=True)
    WH['month'] = WH['date'].dt.month
    WH['weekday'] = WH['date'].dt.weekday
    WH['day'] = WH['date'].dt.day
    WH = WH.dropna()
    if "Temperature (C)" in WH.columns:
        WH = WH.rename(columns={"Temperature (C)": "avg_temp"})

    return WH
