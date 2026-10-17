# Practical 1: Implementation of Basic Python Libraries
# math, statistics and numpy

import math
import statistics
import numpy as np

print("1. MATH LIBRARY DEMONSTRATION")

# Basic math operations
print(f"Square root of 16     : {math.sqrt(16)}")
print(f"Power (2^5)           : {math.pow(2, 5)}")
print(f"Ceiling of 4.3        : {math.ceil(4.3)}")
print(f"Floor of 4.9          : {math.floor(4.9)}")
print(f"Factorial of 5        : {math.factorial(5)}")
print(f"Value of pi           : {math.pi}")
print(f"Value of e            : {math.e}")
print(f"sin(90°) in radians   : {math.sin(math.radians(90))}")
print(f"log10 of 1000         : {math.log10(1000)}")

print("2. STATISTICS LIBRARY DEMONSTRATION")

data = [12, 15, 18, 22, 15, 30, 25, 18, 15, 20]
print(f"Data                  : {data}")
print(f"Mean                  : {statistics.mean(data)}")
print(f"Median                : {statistics.median(data)}")
print(f"Mode                  : {statistics.mode(data)}")
print(f"Variance              : {statistics.variance(data)}")
print(f"Standard Deviation    : {statistics.stdev(data)}")
print(f"Harmonic Mean         : {statistics.harmonic_mean(data)}")

print("3. NUMPY LIBRARY DEMONSTRATION")

# Creating arrays
arr1 = np.array([1, 2, 3, 4, 5])
arr2 = np.array([[1, 2, 3], [4, 5, 6]])
print(f"1D Array              : {arr1}")
print(f"2D Array              :\n{arr2}")

# Array operations
print(f"\nArray + 10            : {arr1 + 10}")
print(f"Array * 2             : {arr1 * 2}")
print(f"Square of array       : {np.square(arr1)}")
print(f"Square root           : {np.sqrt(arr1)}")
print(f"Mean of array         : {np.mean(arr1)}")
print(f"Sum of array          : {np.sum(arr1)}")
print(f"Max value             : {np.max(arr1)}")
print(f"Min value             : {np.min(arr1)}")

# Random numbers
print(f"\nRandom array (0-1)    : {np.random.rand(5)}")
print(f"Random integers       : {np.random.randint(1, 50, 5)}")

# Matrix multiplication
a = np.array([[1, 2], [3, 4]])
b = np.array([[5, 6], [7, 8]])
print(f"\nMatrix A:\n{a}")
print(f"Matrix B:\n{b}")
print(f"Matrix Multiplication (A @ B):\n{a @ b}")