# Practical 4: Linear Regression for Predicting Continuous Values

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# Sample Dataset: Hours Studied vs Marks Scored
X = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]).reshape(-1, 1)
y = np.array([20, 30, 35, 45, 50, 60, 65, 70, 80, 90])

print("Dataset:")
print("Hours Studied (X):", X.flatten())
print("Marks Scored (y):", y)

# Method 1: Using scikit-learn
model = LinearRegression()
model.fit(X, y)

m = model.coef_[0]
c = model.intercept_

print("\nModel Parameters:")
print("Slope (m):", round(m, 4))
print("Intercept (c):", round(c, 4))

# Predictions
y_pred = model.predict(X)
print("\nPredicted Marks:", np.round(y_pred, 2))

# Evaluation
mse = mean_squared_error(y, y_pred)
r2 = r2_score(y, y_pred)

print("\nEvaluation Metrics:")
print("Mean Squared Error (MSE):", round(mse, 4))
print("R2 Score:", round(r2, 4))

# Predict for a new value
hours = 7.5
predicted_marks = model.predict([[hours]])
print(f"\nPredicted marks for {hours} hours of study: {round(predicted_marks[0], 2)}")

# Visualization
plt.scatter(X, y, color='blue', label='Actual Data')
plt.plot(X, y_pred, color='red', label='Regression Line')
plt.xlabel('Hours Studied')
plt.ylabel('Marks Scored')
plt.title('Linear Regression: Hours vs Marks')
plt.legend()
plt.grid(True)
plt.show()