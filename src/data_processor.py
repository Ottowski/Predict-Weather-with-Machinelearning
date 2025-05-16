import pandas as pd

def data_processor(WH):
    WH = WH.rename(columns={'Formatted Date': 'date'})
    WH['date'] = pd.to_datetime(WH['date'], utc=True)
    WH['month'] = WH['date'].dt.month
    WH['weekday'] = WH['date'].dt.weekday
    WH['day'] = WH['date'].dt.day

    print("Columns after processing:", WH.columns.tolist())  

    return WH

if __name__ == "__main__":
    df = pd.read_csv("data/weatherHistory.csv")
    df_clean = data_processor(df)
    print(df_clean.head())
