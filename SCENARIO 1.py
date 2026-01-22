import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv('/Users/jessicasam/Downloads/data.csv',encoding="ISO-8859-1")
print(df.isnull().sum()) 
sales = df.groupby("Description")["Quantity"].sum().head(10)
 
sales.plot(kind="bar", figsize=(10,5))
plt.title("Top 10 Products by Quantity Sold")
plt.xlabel("Product")
plt.ylabel("Quantity")
plt.show()
 
sales.plot(kind="line", marker="o")
plt.title("Sales Trend of Top Products")
plt.xlabel("Product")
plt.ylabel("Quantity")
plt.show()
