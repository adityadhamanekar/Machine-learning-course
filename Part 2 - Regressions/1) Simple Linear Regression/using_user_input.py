import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Importing dataset
dataset = pd.read_csv(r'C:\Users\Dell\Desktop\Machine Learning\course\Part 2 - Regressions\1) Simple Linear Regression\Salary_Data.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

# Splitting dataset into training set and test set
x_train, x_test, y_train, y_test = train_test_split(X, y, train_size=0.2, random_state=0)

# Training the simple linear regression model
regressor = LinearRegression()
regressor.fit(x_train, y_train)

# Predicting the test set result
y_pred = regressor.predict(x_test)

# Making predictions on new input data
new_input = np.array([[5]])  # Example: 5 years of experience
new_prediction = regressor.predict(new_input)

print(f"Predicted salary for 5 years of experience: {new_prediction[0]}")
