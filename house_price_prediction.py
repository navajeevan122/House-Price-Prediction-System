# ==========================================
# HOUSE PRICE PREDICTION PROJECT
# COMPLETE CORRECT CODE
# ==========================================

# Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

# ==========================================
# LOAD DATASET
# ==========================================

# Make sure housing.csv is in same folder
data = pd.read_csv("housing.csv")

print("===== FIRST 5 ROWS =====")
print(data.head())

# ==========================================
# CHECK MISSING VALUES
# ==========================================

print("\n===== MISSING VALUES =====")
print(data.isnull().sum())

# Remove missing values
data = data.dropna()

# ==========================================
# FEATURES AND TARGET
# ==========================================

# Input Features
X = data[[
    "area",
    "bedrooms",
    "bathrooms",
    "stories",
    "parking"
]]

# Target Output
y = data["price"]

# ==========================================
# TRAIN TEST SPLIT
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# ==========================================
# CREATE MODEL
# ==========================================

model = LinearRegression()

# Train Model
model.fit(X_train, y_train)

# ==========================================
# PREDICTIONS
# ==========================================

y_pred = model.predict(X_test)

# ==========================================
# MODEL EVALUATION
# ==========================================

mae = mean_absolute_error(y_test, y_pred)

mse = mean_squared_error(y_test, y_pred)

rmse = np.sqrt(mse)

r2 = r2_score(y_test, y_pred)

print("\n===== MODEL EVALUATION =====")

print(f"MAE        : {mae:.2f}")
print(f"MSE        : {mse:.2f}")
print(f"RMSE       : {rmse:.2f}")
print(f"R2 Score   : {r2:.2f}")

# ==========================================
# MODEL COEFFICIENTS
# ==========================================

print("\n===== FEATURE COEFFICIENTS =====")

coefficients = pd.DataFrame({
    "Feature": X.columns,
    "Coefficient": model.coef_
})

print(coefficients)

# ==========================================
# VISUALIZATION 1
# ACTUAL VS PREDICTED
# ==========================================

plt.figure(figsize=(8, 6))

plt.scatter(y_test, y_pred)

plt.xlabel("Actual Prices")
plt.ylabel("Predicted Prices")

plt.title("Actual vs Predicted House Prices")

plt.grid(True)

plt.show()

# ==========================================
# VISUALIZATION 2
# RESIDUAL ERROR PLOT
# ==========================================

errors = y_test - y_pred

plt.figure(figsize=(8, 6))

plt.scatter(y_pred, errors)

plt.axhline(y=0)

plt.xlabel("Predicted Prices")
plt.ylabel("Residual Errors")

plt.title("Residual Error Plot")

plt.grid(True)

plt.show()

# ==========================================
# VISUALIZATION 3
# FEATURE IMPORTANCE
# ==========================================

plt.figure(figsize=(8, 5))

plt.bar(
    coefficients["Feature"],
    coefficients["Coefficient"]
)

plt.xlabel("Features")
plt.ylabel("Coefficient Value")

plt.title("Feature Importance")

plt.grid(True)

plt.show()

# ==========================================
# CUSTOM HOUSE PRICE PREDICTION
# ==========================================

print("\n===== CUSTOM HOUSE PRICE PREDICTION =====")

try:
    area = float(input("Enter area: "))
    bedrooms = int(input("Enter bedrooms: "))
    bathrooms = int(input("Enter bathrooms: "))
    stories = int(input("Enter stories: "))
    parking = int(input("Enter parking spaces: "))

    # Create input dataframe
    custom_data = pd.DataFrame({
        "area": [area],
        "bedrooms": [bedrooms],
        "bathrooms": [bathrooms],
        "stories": [stories],
        "parking": [parking]
    })

    # Predict Price
    predicted_price = model.predict(custom_data)

    print(
        f"\nPredicted House Price: ₹{predicted_price[0]:,.2f}"
    )

except Exception as e:
    print("\nError:", e)

# ==========================================
# LINEAR REGRESSION FORMULA
# ==========================================

print("\nLinear Regression Formula:")
