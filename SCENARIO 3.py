import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
 
df = pd.read_csv("/Users/jessicasam/Downloads/Housing.csv")
 
print(df.head())
print(df.columns)
print(df.isnull().sum())
 
plt.scatter(df["area"], df["price"])
plt.title("Area vs House Price")
plt.xlabel("Area")
plt.ylabel("Price")
plt.show()
 
numeric_df = df.select_dtypes(include=['int64', 'float64'])
 
plt.figure(figsize=(10,6))
sns.heatmap(numeric_df.corr(), annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap of Numeric Housing Features")
plt.show()
