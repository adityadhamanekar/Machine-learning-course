# Importing the libraries
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression

# Importing the dataset
dataset = pd.read_csv(r'C:\Users\Dell\Desktop\Machine Learning\course\Part 2 - Regressions\2) Multiple Linear Regression\50_Startups.csv')
X = dataset.iloc[:, :-1].values  # Independent variables
y = dataset.iloc[:, -1].values   # Dependent variable

# Function to encode categorical data dynamically
def encode_categorical(data, column_index):
    # Extract the column to encode
    column_data = data[:, column_index]
    
    # Create a dictionary to store unique categories and their indices
    category_index = {category: index for index, category in enumerate(np.unique(column_data))}
    
    # Initialize encoded data with zeros
    encoded_data = np.zeros((len(column_data), len(category_index)), dtype=np.int)
    
    # Encode each entry in the column
    for i, category in enumerate(column_data):
        encoded_data[i, category_index[category]] = 1
    
    return encoded_data, category_index

# Example of how to use the encoding function
state_input = input("Enter State or Country: ")

# Example dataset for demonstration
example_data = np.array(['New York', 'California', 'Florida', 'California', 'New York']).reshape(-1, 1)

# Encode the example data
encoded_example, example_index = encode_categorical(example_data, column_index=0)
print(f"Encoded Example:\n{encoded_example}")
print(f"Category Index:\n{example_index}")

# Continue with the rest of your code...

# Encoding categorical data (state column in this case)
# You can replace '3' with the index of the column containing states or countries in your dataset
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [3])], remainder='passthrough')
X = np.array(ct.fit_transform(X))

# Splitting the dataset into the Training set and Test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Training the Multiple Linear Regression model on the Training set
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Predicting the Test set results
y_pred = regressor.predict(X_test)

# Function to predict profit based on user input with generalized encoding
def predict_startup_profit(state_input, RnD_Spend, Administration, Marketing_Spend):
    # Encode the state input using the example index
    encoded_state = np.zeros((1, len(example_index)))
    if state_input in example_index:
        encoded_state[0, example_index[state_input]] = 1
    else:
        print("Category not found in example index. Defaulting to all zeros.")
    
    # Prepare input data
    user_input = np.hstack((encoded_state, [[RnD_Spend, Administration, Marketing_Spend]]))

    # Predict the profit
    predicted_profit = regressor.predict(user_input)
    return predicted_profit[0]

# Example usage of the prediction function
state_input = input("Enter State or Country: ")
RnD_input = float(input("Enter R&D Spend: "))
Admin_input = float(input("Enter Administration Spend: "))
Marketing_input = float(input("Enter Marketing Spend: "))

predicted_profit = predict_startup_profit(state_input, RnD_input, Admin_input, Marketing_input)
print(f"Predicted Profit: ${predicted_profit:.2f}")
