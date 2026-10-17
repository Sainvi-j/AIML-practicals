# Practical 2: Pandas and Matplotlib for ML Applications

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


print("1. PANDAS DEMONSTRATION")

# Creating a sample dataset (Student Marks)
data = {
    'Name': ['Amit', 'Priya', 'Rahul', 'Sneha', 'Vikram', 'Neha', 'Rohan', 'Anjali'],
    'Maths': [78, 85, 62, 91, 55, 88, 73, 95],
    'Science': [82, 79, 68, 88, 60, 92, 70, 90],
    'English': [75, 88, 71, 85, 65, 80, 78, 93],
    'Age': [18, 19, 18, 20, 19, 18, 20, 19]
}

df = pd.DataFrame(data)

print("\n Original DataFrame ")
print(df)

print("\n First 5 rows (head) ")
print(df.head())

print("\n Basic Information")
print(df.info())

print("\n Statistical Summary")
print(df.describe())

print("\n Column Names")
print(df.columns.tolist())

print("\n Shape of DataFrame")
print(df.shape)

print("\n Selecting specific columns ")
print(df[['Name', 'Maths', 'Science']])

print("\n Filtering: Students with Maths > 80")
print(df[df['Maths'] > 80])

print("\n Adding a new column: Total Marks")
df['Total'] = df['Maths'] + df['Science'] + df['English']
print(df)

print("\n Sorting by Total Marks (Descending) ")
print(df.sort_values(by='Total', ascending=False))

print("\n Group by Age and calculate mean marks ")
print(df.groupby('Age')[['Maths', 'Science', 'English']].mean())

print("\n Checking for missing values ")
print(df.isnull().sum())

print("2. MATPLOTLIB VISUALIZATION")

# Set style for better looking plots
plt.style.use('seaborn-v0_8-darkgrid')

#  Plot 1: Bar Chart - Total Marks of Students 
plt.figure(figsize=(10, 5))
plt.bar(df['Name'], df['Total'], color='skyblue', edgecolor='black')
plt.title('Total Marks of Students', fontsize=14)
plt.xlabel('Student Name')
plt.ylabel('Total Marks')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

#  Plot 2: Line Chart - Subject-wise Marks 
plt.figure(figsize=(10, 5))
plt.plot(df['Name'], df['Maths'], marker='o', label='Maths', linewidth=2)
plt.plot(df['Name'], df['Science'], marker='s', label='Science', linewidth=2)
plt.plot(df['Name'], df['English'], marker='^', label='English', linewidth=2)
plt.title('Subject-wise Marks Comparison', fontsize=14)
plt.xlabel('Student Name')
plt.ylabel('Marks')
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Plot 3: Scatter Plot - Maths vs Science 
plt.figure(figsize=(8, 5))
plt.scatter(df['Maths'], df['Science'], color='green', s=100, alpha=0.7)
plt.title('Maths vs Science Marks', fontsize=14)
plt.xlabel('Maths Marks')
plt.ylabel('Science Marks')
for i, name in enumerate(df['Name']):
    plt.annotate(name, (df['Maths'][i], df['Science'][i]), fontsize=8)
plt.tight_layout()
plt.show()

# Plot 4: Histogram - Distribution of Total Marks
plt.figure(figsize=(8, 5))
plt.hist(df['Total'], bins=5, color='orange', edgecolor='black', alpha=0.7)
plt.title('Distribution of Total Marks', fontsize=14)
plt.xlabel('Total Marks')
plt.ylabel('Number of Students')
plt.tight_layout()
plt.show()

# Plot 5: Pie Chart - Average Marks per Subject
avg_marks = [df['Maths'].mean(), df['Science'].mean(), df['English'].mean()]
subjects = ['Maths', 'Science', 'English']
colors = ['#ff9999', '#66b3ff', '#99ff99']

plt.figure(figsize=(7, 7))
plt.pie(avg_marks, labels=subjects, colors=colors, autopct='%1.1f%%',
        startangle=90, explode=(0.05, 0.05, 0.05))
plt.title('Average Marks Distribution by Subject', fontsize=14)
plt.tight_layout()
plt.show()

print("\nAll plots have been displayed successfully!")