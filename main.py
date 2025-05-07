import pandas as pd
from src.data_loader import load_weather_data

def main():
    # Reads data
    WH = load_weather_data("data/weatherHistory.csv")

    print("\n Modell is trained and ready for use!")

if __name__ == "__main__":
    main()
