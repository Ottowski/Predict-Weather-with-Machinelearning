import pandas as pd
from src.data_loader import load_weather_data
from src.data_model import train_linear_regression
from src.data_processor import data_processor

def main():
    # Reads data
    WH = load_weather_data("data/weatherHistory.csv")
    print(WH.columns)
    
    # Treat data
    WH_clean = data_processor(WH)
    
    # Trains data
    Model = train_linear_regression(WH_clean)

    print("\n Modell is trained and ready for use!")

if __name__ == "__main__":
    main()
