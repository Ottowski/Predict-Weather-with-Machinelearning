from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import root_mean_squared_error
import pandas as pd
import joblib

# Load data
WH = pd.read_csv(r"data/weatherHistory.csv")
WH = WH.rename(columns={'Formatted Date': 'date'})
WH['date'] = pd.to_datetime(WH['date'], utc=True)
WH = WH.dropna()

# Filter
WH_filtered = WH[
    (WH['Temperature (C)'] > -40) & (WH['Temperature (C)'] < 60) &
    (WH['Pressure (millibars)'] > 970) & (WH['Pressure (millibars)'] < 1050) &
    (WH['Humidity'] >= 0) & (WH['Humidity'] <= 1.5) &
    (WH['Wind Speed (km/h)'] >= 0) & (WH['Wind Speed (km/h)'] <= 80)
]

# Feature and target
X = WH_filtered[['Temperature (C)', 'Wind Speed (km/h)', 'Pressure (millibars)']]
y = WH_filtered['Humidity']

# Train
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
rmse = root_mean_squared_error(y_test, y_pred)
print(f"Humidity Model RMSE: {rmse:.4f}")

# Save
joblib.dump(model, 'saved-models/humidity_model.pkl')
print("Model saved: saved-models/humidity_model.pkl")