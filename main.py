import pandas as pd
from src.data_loader import load_weather_data
from src.data_model import train_linear_regression
from src.data_processor import data_processor
import seaborn as sns
import matplotlib.pyplot as plt

def main():
    # Reads data
    WH = load_weather_data("data/weatherHistory.csv")
    print(WH.columns)
    
    # Data Cleaning
    WH_clean = data_processor(WH)
    
    # Trains data
    Model = train_linear_regression(WH_clean)
    
    print("\n Modell is trained and ready for use!")

    # Check if 'Temperature (C)' exists in the data and plot pairplot
    if 'Temperature (C)' in WH.columns:
        sns.pairplot(WH[['Temperature (C)', 'Humidity', 'Wind Speed (km/h)', 'Pressure (millibars)']])
        plt.show()
    else:
        print("Kolumnen 'Temperature (C)' finns inte i datan.")

if __name__ == "__main__":
    main()
