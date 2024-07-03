# importing libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt



# importing dataset
dataset = pd.read_csv(r'C:\Users\Dell\Desktop\Machine Learning\course\Part 2 - Regressions\1) Simple Linear Regression\Salary_Data.csv')
X = dataset.iloc[:, : -1].values
y = dataset.iloc[:, -1].values


# Splitting dataset in training set adn test set
from sklearn.model_selection import train_test_split
x_train , x_test , y_train , y_test = train_test_split(X, y , test_size=0.2, random_state=0)



# training simple regression model 
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train, y_train)


# predicting the test set result
y_pred = regressor.predict(x_test)


# visualizing result for training set
plt.scatter(x_train, y_train, color='red')
plt.plot(x_train, regressor.predict(x_train), color='blue')
plt.title("salary vs year of experiance of training set")
plt.xlabel('no of year')
plt.ylabel('salary')
plt.show()



# visualizing result for test set
plt.scatter(x_test, y_test, color='red')
plt.plot(x_train, regressor.predict(x_train), color='blue')
plt.title("salary vs year of experiance of training set")
plt.xlabel('no of year')
plt.ylabel('salary')
plt.show()
