# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

def load_data(file_path):
    """
    Load the dataset from a CSV file.

    Parameters:
        file_path (str): Path to the CSV file.

    Returns:
        pd.DataFrame: Loaded dataset.
    """
    data = pd.read_csv(file_path)
    return data

def perform_eda(data):
    """
    Perform Exploratory Data Analysis (EDA) on the dataset.

    Parameters:
        data (pd.DataFrame): Input dataset.

    Returns:
        None
    """
    print("Exploratory Data Analysis:")
    print(data.info())
    print(data.describe())

def prepare_data(data):
    """
    Prepare the dataset by selecting relevant features and target variable.

    Parameters:
        data (pd.DataFrame): Input dataset.

    Returns:
        pd.DataFrame, np.array: Features (X) and Labels (y).
    """
    data = data[["G1", "G2", "G3", "studytime", "failures", "absences"]]
    predict = "G3"
    X = np.array(data.drop(columns=[predict]))  # Features
    y = np.array(data[predict])  # Labels
    return X, y

def train_linear_model(X_train, y_train):
    """
    Train a linear regression model.

    Parameters:
        X_train (np.array): Features for training.
        y_train (np.array): Labels for training.

    Returns:
        linear_model.LinearRegression: Trained linear regression model.
    """
    linear = linear_model.LinearRegression()
    linear.fit(X_train, y_train)
    return linear

def evaluate_model(model, X_test, y_test):
    """
    Evaluate the linear regression model.

    Parameters:
        model (linear_model.LinearRegression): Trained linear regression model.
        X_test (np.array): Features for testing.
        y_test (np.array): Labels for testing.

    Returns:
        dict: Dictionary of evaluation metrics.
    """
    y_pred = model.predict(X_test)
    metrics = {
        'R-squared (R²)': r2_score(y_test, y_pred),
        'Mean Absolute Error (MAE)': mean_absolute_error(y_test, y_pred),
        'Mean Squared Error (MSE)': mean_squared_error(y_test, y_pred),
        'Root Mean Squared Error (RMSE)': np.sqrt(mean_squared_error(y_test, y_pred))
    }
    return metrics

if __name__ == "__main__":
    # Load data
    dataset = load_data("student.csv")

    # Perform EDA
    perform_eda(dataset)

    # Prepare data
    X, y = prepare_data(dataset)

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)

    # Train the model
    model = train_linear_model(X_train, y_train)

    # Evaluate the model
    evaluation_metrics = evaluate_model(model, X_test, y_test)
    for metric, value in evaluation_metrics.items():
        print(f"{metric}: {value}")
    
    # Display coefficients and intercept
    print('Coefficient (Slope Values): ', model.coef_)
    print('Intercept: ', model.intercept_)