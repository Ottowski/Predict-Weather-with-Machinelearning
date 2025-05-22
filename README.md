# Predict Weather with Machine Learning

This project uses **machine learning models** to predict key weather attributes such as **humidity**, **temperature**, **wind speed**, and **air pressure** based on historical weather data (csv file). It's built using Python and scikit-learn, with a pipeline that includes data preprocessing, training, saving, and running predictions.

##  Requirements
How to Install dependencies using pip:

bash:
pip install pandas matplotlib seaborn scikit-learn

## How to Run

Make sure you have the required Python packages installed (see below), then run the following scripts in this order:

##  1. Preprocess and Save Models
Trains and saves all models into the saved-models/ folder:
bash:
python save_models.py

Output:
Good! The file is read succesfully.
Columns after processing: [...]
All models have been trained and saved in the 'saved-models/' folder.

##  2. Train Additional Prediction Models
Stores separate models in predict-models/:

bash:
python train_predict_models.py

Output:
All models are trained and saved in the 'predict-models/' folder.

##  3. Run Example Predictions
Tests all models with sample input values:
bash:
python run_all_predictions.py

Sample Output:
Predicted Humidity: 0.600
Predicted Wind Speed: 8.89 km/h
Predicted Pressure: 1001.42 millibars
Predicted Temperature: 11.50 °C

##  4. Explore and Analyze the Dataset
Displays thorugh seaborn graphs: dataset shape, column info, summary statistics, and missing values.
bash:
python main.py

Shows info, stats, and missing values:

lua
Dataframe shape: (96453, 12)
--- Info ---
<class 'pandas.core.frame.DataFrame'> ...
--- Description ---
Summary statistics...
--- Missing values ---
Precip Type    517

##  Features & Capabilities
Cleans and processes real weather data.
Predicts multiple weather metrics using ML regressors.
Adds time-based features (month, weekday, day).
Handles missing values gracefully.
Trains and stores models for future reuse.
Modular and maintainable Python codebase.


##  Machine Learning Models Used
Random Forest Regressor
Linear Regression


##  Evaluation Metrics
Each model is evaluated using:
RMSE – Root Mean Squared Error
MAE – Mean Absolute Error
R² Score – Coefficient of Determination

##  Example Prediction Input/Output
Input:

temperature = 22.5
humidity = 0.6
wind_speed = 10.5
pressure = 1012.0

Output:

Predicted Humidity: 0.6
Predicted Wind Speed: 10.2 km/h
Predicted Pressure: 1010.4 mb
Predicted Temperature: 21.9 °C

##  Processed Data Columns
After cleaning, key columns used for modeling:

Temperature (C)
Humidity
Wind Speed (km/h)
Pressure (millibars)
Apparent Temperature (C)
Wind Bearing (degrees)
Visibility (km)
Precip Type
Summary
Daily Summary
Extracted time features: month, weekday, day

##  Future Improvements
Add GUI or dashboard using Tkinter or Streamlit
Deploy as a REST API with FastAPI or Flask
Add real-time data fetching
Predict more attributes (e.g., precipitation chance)

## Author
Made by Otto Arvidsson, 2025