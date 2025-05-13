import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Read Data
df = pd.read_csv(r"C:\Users\ottoa\OneDrive\Skrivbord\Predict Weather with Machinelearning\data\weatherHistory.csv")

# Data Cleaning
df = df.rename(columns={'Formatted Date': 'date'})
df['date'] = pd.to_datetime(df['date'], utc=True)
df['month'] = df['date'].dt.month
df['weekday'] = df['date'].dt.weekday
df['day'] = df['date'].dt.day
df = df.dropna()

# Renames Temperature
if "Temperature (C)" in df.columns:
    df = df.rename(columns={"Temperature (C)": "avg_temp"})

# Show Columns
print("Columns in pairplot:")
print(df[['avg_temp', 'Humidity', 'Wind Speed (km/h)', 'Pressure (millibars)']].head())

# Visualize with Seaborn pairplot
sns.pairplot(df[['avg_temp', 'Humidity', 'Wind Speed (km/h)', 'Pressure (millibars)']])
plt.suptitle("Pairplot av väderparametrar", y=1.02)
plt.show()