import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Exploratory Data Analysis (EDA) class to provide an overview analysis of a pandas DataFrame through summarization, null-checking, histogram, and correlation map.
class EDA:
    def __init__(self, WH: pd.DataFrame):
        self.WH = WH
    
    def summary(self):
        # Prints the size of the dataset, column info, and descriptive statistics.
        print("Dataframe shape:", self.WH.shape)
        print("\n--- Info ---")
        self.WH.info()
        print("\n--- Description ---")
        print(self.WH.describe(include='all'))
    
    def check_nulls(self):
        # Checking potentiel missing values
        print("\n--- Missing values ---")
        nulls = self.WH.isnull().sum()
        print(nulls[nulls > 0] if nulls.sum() > 0 else "No missing values found.")
    
    def plot_histograms(self, cols=None, bins=30):
        # Plots a histograms for numeric columns. Parameters:
        # cols: List of columns to include (default: all numeric)
        # bins: Number of bins in the histogram
        if cols is None:
            cols = self.WH.select_dtypes(include=['float64', 'int64']).columns
        self.WH[cols].hist(bins=bins, figsize=(15, 10))
        plt.tight_layout()
        plt.show()
    
    def plot_correlation_heatmap(self):
        # Plots a correlation map (heatmap) for numeric columns in the dataset.
        corr = self.WH.corr(numeric_only=True)
        plt.figure(figsize=(10, 8))
        sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f")
        plt.title("Correlation heatmap")
        plt.show()
