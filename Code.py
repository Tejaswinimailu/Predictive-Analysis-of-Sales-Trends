        # Predictive Analysis of Sales Trends 

# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Load the sales data (Assuming you have a CSV file with columns like 'Date' and 'Sales')
data = pd.read_csv('sales_tarin.csv')

# Explore the data
print(data.head())

# Data Preprocessing
# You might need to handle missing values, convert 'Date' to datetime, and perform feature engineering.

# Extract features and target variable
X = data[['feature1', 'feature2', ...]]  # Add relevant features
y = data['Sales']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train a predictive model (Linear Regression as an example)
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse}")
print(f"R-squared: {r2}")

# Visualize the results (you can use more sophisticated plots for actual analysis)
plt.scatter(X_test, y_test, color='black')
plt.plot(X_test, y_pred, color='blue', linewidth=3)
plt.xlabel('Features')
plt.ylabel('Sales')
plt.title('Sales Prediction')
plt.show()

# Make predictions for future sales
future_data = pd.read_csv('future_data.csv')  # Load data for future predictions
future_X = future_data[['feature1', 'feature2', ...]]  # Use the same features
future_predictions = model.predict(future_X)

# Visualize future predictions
plt.plot(future_data['Date'], future_predictions, color='green', marker='o')
plt.xlabel('Date')
plt.ylabel('Predicted Sales')
plt.title('Future Sales Predictions')
plt.xticks(rotation=45)
plt.show()
