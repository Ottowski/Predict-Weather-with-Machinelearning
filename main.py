from src.data_loader import load_weather_data
from src.data_model import train_linear_regression
from src.data_processor import data_processor
from src.random_forest_model import train_random_forest
from sklearn.model_selection import train_test_split
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def main():
    # Reads data
    WH = load_weather_data("data/weatherHistory.csv")
    print("Original Columns:")
    print(WH.columns)

    # Data Cleaning
    WH_clean = data_processor(WH)
    print("Cleaned data:")
    print(WH_clean.head())

    # Control if 'avg_temp' already exist
    if "avg_temp" not in WH_clean.columns:
        print("Column 'avg_temp' doesn't exist in the cleaned data.")
        return
    
    # Create features and target
    X = WH_clean.drop(["avg_temp", "date", "Summary", "Precip Type", "Daily Summary"], axis=1)
    y = WH_clean["avg_temp"]

    # Split up trained and test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Trains data
    model = train_linear_regression(WH_clean)
    print("\nModell is trained and ready for use!")

    # Train Random Forest
    rf_model = train_random_forest(X_train, X_test, y_train, y_test)

    # Filter data to realistic value ranges
    WH_filtered = WH[
        (WH['Temperature (C)'] > -40) & (WH['Temperature (C)'] < 60) &
        (WH['Pressure (millibars)'] > 970) & (WH['Pressure (millibars)'] < 1050) &
        (WH['Humidity'] >= -2.0) & (WH['Humidity'] <= 4.0) &
        (WH['Wind Speed (km/h)'] >= 0) & (WH['Wind Speed (km/h)'] <= 100)
    ]

    # Check if 'Temperature (C)' exists in the data and plot pairplot
    if 'Temperature (C)' in WH.columns:
        # Create pairplot to better visualize features
        sns.pairplot(WH_filtered[['Temperature (C)', 'Humidity', 'Wind Speed (km/h)', 'Pressure (millibars)']])
        plt.suptitle("Filtered Feature Relationships (Expanded Ranges)", y=1.02)
        plt.show()
    else:
        print("Column 'Temperature (C)' doesn't exist in the cleaned data.")

if __name__ == "__main__":
    main()
