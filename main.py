import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import missingno as msno

msno.matrix(df)
plt.show()

df = pd.read_csv("weatherHistory.csv")  

print(df.head())

print("\nNull values in each column:")
print(df.isnull().sum())

print("\nDescriptive statistics:")
print(df.describe())

print("\nData types in the dataset:")
print(df.dtypes)


df['date'] = pd.to_datetime(df['date'])
df['month'] = df['date'].dt.month
df['weekday'] = df['date'].dt.weekday
df['day'] = df['date'].dt.day



df['avg_temp'].hist(bins=20)
plt.title('Histogram of average temperature')
plt.xlabel('Temperature')
plt.ylabel('Days')
plt.show()

plt.figure(figsize=(10, 8))
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap='coolwarm')
plt.title('Correlationheatmap')
plt.show()


sns.scatterplot(data=df, x='humidity', y='avg_temp')
plt.title('Samband mellan luftfuktighet och medeltemperatur')
plt.show()
