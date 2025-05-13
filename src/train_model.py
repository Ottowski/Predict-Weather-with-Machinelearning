import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import joblib

# Load data
df = pd.read_csv(r"data/weatherHistory.csv")
df = df.rename(columns={'Formatted Date': 'date', 'Temperature (C)': 'avg_temp'})
df['date'] = pd.to_datetime(df['date'], utc=True)
df = df.dropna()

# Feature and target
X = df[['Humidity', 'Wind Speed (km/h)', 'Pressure (millibars)']]
y = df['avg_temp']

# Split train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Evaluate model
y_pred = model.predict(X_test)
rmse = mean_squared_error(y_test, y_pred, squared=False)
print(f"Model trained. RMSE: {rmse:.2f}")

# Save model
joblib.dump(model, 'model/linear_model.pkl')
print("Model saved to model/linear_model.pkl'")
