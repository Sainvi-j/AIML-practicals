# Practical 9: Mechatronics Application using ML
# Predictive Maintenance of DC Motor

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# Dataset: [Current, Speed, Vibration, Operating_Hours] -> Temperature
X = np.array([
    [2.5, 1200, 0.3, 1],
    [3.0, 1400, 0.4, 2],
    [3.5, 1600, 0.5, 3],
    [4.0, 1800, 0.6, 4],
    [4.5, 2000, 0.7, 5],
    [5.0, 2200, 0.8, 6],
    [5.5, 2400, 0.9, 7],
    [6.0, 2600, 1.0, 8],
    [2.8, 1300, 0.35, 1.5],
    [3.2, 1500, 0.45, 2.5],
    [3.8, 1700, 0.55, 3.5],
    [4.2, 1900, 0.65, 4.5],
    [4.8, 2100, 0.75, 5.5],
    [5.2, 2300, 0.85, 6.5],
    [5.8, 2500, 0.95, 7.5]
])

y = np.array([45, 52, 58, 65, 72, 80, 88, 95, 48, 55, 62, 69, 76, 84, 91])

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Evaluation
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Mean Squared Error (MSE):", round(mse, 4))
print("R2 Score (Accuracy):", round(r2, 4))
print("Accuracy in percentage:", round(r2 * 100, 2), "%")

print("\nActual Temperature   :", y_test)
print("Predicted Temperature:", np.round(y_pred, 2))

# Predict temperature for a new motor condition
new_data = np.array([[4.7, 2050, 0.72, 5.2]])
predicted_temp = model.predict(new_data)
print("\nPredicted Temperature for new condition:", round(predicted_temp[0], 2), "°C")

# Visualization
plt.scatter(y_test, y_pred, color='blue')
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='--')
plt.xlabel('Actual Temperature (°C)')
plt.ylabel('Predicted Temperature (°C)')
plt.title('Actual vs Predicted Motor Temperature')
plt.grid(True)
plt.show()