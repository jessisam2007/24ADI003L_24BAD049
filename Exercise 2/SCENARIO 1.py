print("JESSICA SAM - 24BAD049")
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error, r2_score

df = pd.read_csv("/Users/jessicasam/Downloads/bottle.csv",encoding="ISO-8859-1",low_memory=False)
print(df.head())
features = ['Depthm', 'Salnty', 'O2ml_L','STheta', 'O2Sat']
target = 'T_degC'
df = df[features + [target]]
print(df.info())

df.fillna(df.mean(), inplace=True)
print("\nMissing values after imputation:")
print(df.isnull().sum())

X = df[features]
y = df[target]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42)
lr = LinearRegression()
lr.fit(X_train, y_train)
y_pred = lr.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)
print("\nModel Performance:")
print("MSE :", mse)
print("RMSE:", rmse)
print("R²  :", r2)

plt.figure(figsize=(6,6))
plt.scatter(y_test, y_pred, alpha=0.5)
plt.xlabel("Actual Temperature (°C)")
plt.ylabel("Predicted Temperature (°C)")
plt.title("Actual vs Predicted Water Temperature")
plt.show()

residuals = y_test - y_pred
plt.figure(figsize=(6,4))
sns.histplot(residuals, kde=True)
plt.xlabel("Residual Error")
plt.title("Residual Errors Distribution")
plt.show()

ridge = Ridge(alpha=1.0)
ridge.fit(X_train, y_train)
ridge_pred = ridge.predict(X_test)

print("\nRidge Regression R²:", r2_score(y_test, ridge_pred))

lasso = Lasso(alpha=0.01)
lasso.fit(X_train, y_train)
lasso_pred = lasso.predict(X_test)

print("Lasso Regression R²:", r2_score(y_test, lasso_pred))

coefficients = pd.Series(lr.coef_, index=features)
print("\nFeature Importance (Linear Regression):")
print(coefficients.sort_values(ascending=False))
