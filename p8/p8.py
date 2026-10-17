# Practical 8: ML based Robotic Arm for Pick-and-Drop Operations

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# Sample dataset: Object features
# Features: [Size, Weight, Color_Intensity, Shape_Score]
# Label: 1 = Pickable, 0 = Non-Pickable

X = np.array([
    [5.2, 0.8, 120, 0.9],
    [3.1, 0.3, 80, 0.4],
    [6.0, 1.2, 150, 0.95],
    [2.5, 0.2, 60, 0.3],
    [5.8, 1.0, 140, 0.92],
    [3.5, 0.4, 90, 0.5],
    [6.5, 1.5, 160, 0.98],
    [2.8, 0.25, 70, 0.35],
    [5.5, 0.9, 130, 0.88],
    [3.0, 0.35, 85, 0.45],
    [6.2, 1.3, 155, 0.96],
    [2.2, 0.15, 55, 0.25]
])

y = np.array([1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0])

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train ML Model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Model Accuracy:", round(accuracy * 100, 2), "%")

# Simulation of Robotic Arm
def robotic_arm_pick_and_drop(object_features):
    prediction = model.predict([object_features])[0]
    
    if prediction == 1:
        # Predict target coordinates (simple logic based on size and weight)
        pick_x = round(object_features[0] * 10, 2)
        pick_y = round(object_features[1] * 20, 2)
        drop_x = 50
        drop_y = 50
        
        print("\nObject Status: Pickable")
        print(f"Moving arm to Pick Position: ({pick_x}, {pick_y})")
        print("Gripper Closed - Object Picked")
        print(f"Moving arm to Drop Position: ({drop_x}, {drop_y})")
        print("Gripper Opened - Object Dropped Successfully")
        return "Success"
    else:
        print("\nObject Status: Non-Pickable")
        print("Robotic Arm remains idle.")
        return "Ignored"

# Test with new objects
print("\nTest Case 1 ---")
obj1 = [5.7, 1.1, 145, 0.93]
robotic_arm_pick_and_drop(obj1)

print("\nTest Case 2 ---")
obj2 = [2.9, 0.28, 75, 0.38]
robotic_arm_pick_and_drop(obj2)

# Visualization of Decision
plt.figure(figsize=(7, 5))
plt.scatter(X[y==1][:, 0], X[y==1][:, 1], color='green', label='Pickable', s=80)
plt.scatter(X[y==0][:, 0], X[y==0][:, 1], color='red', label='Non-Pickable', s=80)
plt.xlabel('Object Size')
plt.ylabel('Object Weight')
plt.title('Object Classification for Robotic Arm')
plt.legend()
plt.grid(True)
plt.show()