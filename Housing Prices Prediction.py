
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error


# CREATE SYNTHETIC DATA


np.random.seed(42)

n = 1000

# House size in square meters
size = np.random.normal(120, 35, n)

# Number of rooms
rooms = np.random.randint(1, 7, n)

# House age
age = np.random.randint(0, 50, n)

# Distance from city center (km)
distance = np.random.normal(8, 4, n)


# HOUSE PRICE FORMULA

price = (
    50000
    + 2500 * size
    + 15000 * rooms
    - 1800 * age
    - 4000 * distance
    + np.random.normal(0, 20000, n)
)

# --------------------------------------------------
# CREATE DATAFRAME
# --------------------------------------------------

df = pd.DataFrame({
    'size': size,
    'rooms': rooms,
    'age': age,
    'distance': distance,
    'price': price
})

print(df.head())


# FEATURES AND TARGET


X = df[['size', 'rooms', 'age', 'distance']]
y = df['price']


# TRAIN / TEST SPLIT


X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# --------------------------------------------------
# CREATE MODEL
# --------------------------------------------------

model = LinearRegression()

# --------------------------------------------------
# TRAINING
# --------------------------------------------------

model.fit(X_train, y_train)

# --------------------------------------------------
# PREDICTIONS
# --------------------------------------------------

predictions = model.predict(X_test)

# --------------------------------------------------
# EVALUATION
# --------------------------------------------------

mae = mean_absolute_error(y_test, predictions)

print("\nMean Absolute Error:", round(mae, 2))

# --------------------------------------------------
# COEFFICIENTS
# --------------------------------------------------

coefficients = pd.DataFrame({
    'Feature': X.columns,
    'Coefficient': model.coef_
})

print("\nLearned Coefficients")
print(coefficients)

print("\nIntercept:", model.intercept_)

# --------------------------------------------------
# VISUALIZATION
# --------------------------------------------------

plt.figure(figsize=(10,5))

plt.plot(y_test.values[:100], label='Real Prices')
plt.plot(predictions[:100], label='Predicted Prices')

plt.xlabel('House Sample')
plt.ylabel('House Price')
plt.title('Real vs Predicted House Prices')

plt.legend()
plt.grid(True)

plt.show()


