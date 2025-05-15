from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import root_mean_squared_error
import pandas as pd
import joblib

# Load data
WH = pd.read_csv(r"data/weatherHistory.csv")
WH = WH.rename(columns={'Formatted Date': 'date', 'Temperature (C)': 'avg_temp'})
WH['date'] = pd.to_datetime(WH['date'], utc=True)
WH = WH.dropna()

# Feature and target
X = WH[['Humidity', 'Wind Speed (km/h)', 'Pressure (millibars)']]
y = WH['avg_temp']

# Split train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Evaluate model
y_pred = model.predict(X_test)
rmse = root_mean_squared_error(y_test, y_pred, squared=False)
print(f"Model trained. RMSE: {rmse:.2f}")

# Save model
joblib.dump(model, 'model/linear_model.pkl')
print("Model saved to model/linear_model.pkl'")
