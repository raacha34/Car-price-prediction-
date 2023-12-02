# Car-price-prediction
# Data selection, Data Cleaning, MVP for the project.
 ## Access the app here https://jule2rbgqspegmhvlsknvy.streamlit.app/

Abstract

This project aims to build a web application that predicts the MSRP of a car based on its features using Ridge Regression. The project involved data extraction, cleaning, model building, deployment to a Streamlit app, and pickle file utilization.

Data Description

The data used for this project comes from the Edmunds and Twitter dataset, which includes information such as:

    **Make**: The manufacturer of the car (e.g., Ford, Toyota, Honda)
    **Model**: The specific model of the car (e.g., Mustang, Camry, Civic)
    **Year**: The year the car was manufactured
    **Engine Typ**e: The type of engine the car has (e.g., V6, V8, Hybrid)
    **MSRP**: The Manufacturer's Suggested Retail Price of the car

The data was pre-processed to address missing values, outliers, and categorical features. One-hot encoding was used to transform categorical features into numerical representations suitable for the model.
Algorithm Description

The project utilizes Ridge Regression, a linear regression model with L2 regularization, to predict the MSRP of a car based on its features. The L2 regularization helps to prevent overfitting and improve the model'sgeneralizability.

Here are the steps involved in the algorithm:

    **Data Preparation**: The data is extracted, cleaned, and pre-processed, including missing value imputation, outlier detection and removal, and feature scaling.
    **One-Hot Encoding**: Categorical features are transformed into numerical representations using one-hot encoding.
    **Model Training**: The Ridge Regression model is trained on the prepared data. The model parameters (weights and biases) are learned to minimize the difference between the predicted and actual MSRP values.
    **Prediction**: Once the model is trained, it can be used to predict the MSRP of new car instances based on their features.

**Tools Used**

The project utilizes various tools for different purposes:

    Python: A general-purpose programming language used for data analysis, modeling, and web development.
    Pandas: A Python library for data manipulation and analysis.
    NumPy: A Python library for scientific computing, including linear algebra and array operations.
    Scikit-learn: A Python library for machine learning, including Ridge Regression implementation.
    Streamlit: A Python library for building web applications.
    Pickle: A Python library for serializing and deserializing Python objects, used to save the trained model as a pickle file for deployment.

Here's a breakdown of the specific tools used and their purposes:

    Python: Used for the entire data processing, model building, and web app development process.
    Pandas: Used for data loading, cleaning, manipulation, and exploration.
    NumPy: Used for efficient numerical computations and linear algebra operations.
    Scikit-learn: Used for Ridge Regression model implementation, training, and evaluation.
    Streamlit: Used to build the web application interface for users to interact with the model and obtain predictions.
    Pickle: Used to save the trained Ridge Regression model as a pickle file for easy deployment and use in the web app.

Overall, this project demonstrates the effectiveness of machine learning, specifically Ridge Regression, in predicting car prices. The project also showcases the application of various tools and techniques for data analysis, modeling, and web development. 
