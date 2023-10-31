# Student Grades Prediction Project

## Overview
This project aims to predict a student's final grade (G3) based on various factors such as midterm grades (G1 and G2), study time, failures, and absences using  a dataset from Kaggle. Initially, a Linear Regression model was used for prediction, and later, a Random Forest model was implemented for potentially improved performance.

## Files and Directories
- `pyproject.toml`: Configuration file for project dependencies and settings.
- `student.csv`: Dataset containing student information and grades.
- `student_grades_prediction_linear_regression.ipynb`: Jupyter notebook for Linear Regression model implementation and evaluation.
- `student_grades_prediction_linear_regression.py`: Python script for Linear Regression model implementation.
- `student_grades_prediction_random_forest.ipynb`: Jupyter notebook for Random Forest model implementation and evaluation.
- `student_grades_prediction_random_forest.py`: Python script for Random Forest model implementation.

## Getting Started
1. Install Poetry for dependency management: [Poetry Installation Guide](https://python-poetry.org/docs/#installation).
2. Run `poetry install` to install the project dependencies.
3. Ensure you have the necessary dataset (`student.csv`) in the project directory.

## Dependencies
- `pandas`: Data manipulation and analysis library in Python.
- `numpy`: Numerical computing library for handling arrays and matrices.
- `scikit-learn`: Machine learning library for building and evaluating models.

## Notes
- The Linear Regression model is a statistical approach to modeling the relationship between the independent variables and the target variable. It assumes a linear relationship.
- The Random Forest model is an ensemble learning method that uses multiple decision trees to make predictions.
- Both models are evaluated using metrics including R-squared, Mean Absolute Error (MAE), Mean Squared Error (MSE), and Root Mean Squared Error (RMSE).