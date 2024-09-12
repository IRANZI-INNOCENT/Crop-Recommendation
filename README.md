# Crop Recommendation System

## Overview

This project is a machine learning-based web application that recommends the most suitable crop to cultivate based on various environmental factors such as soil nutrients, temperature, humidity, and rainfall. The web application provides an intuitive interface where users can input environmental parameters, and the system will recommend the most appropriate crop using trained models such as Logistic Regression, Decision Trees, Support Vector Machines, and Random Forest.

## Project Structure

- **app.py**: The main file that runs the web-based interface using **Streamlit**. It allows users to input environmental factors and select a machine learning model for crop recommendation.
  
- **app_functions.py**: Contains the functions for loading the dataset, training, saving, and loading machine learning models. It also includes the recommendation logic that uses the selected model to predict the best crop for cultivation.

- **requirements.txt**: A file listing the required dependencies to run the project.

## Features

- **Model Selection**: Users can choose between four machine learning models:
  - Logistic Regression
  - Decision Tree
  - Support Vector Machine (SVM)
  - Random Forest
  
- **Environmental Factors Input**: Users can provide specific values for soil nutrients (Nitrogen, Phosphorus, Potassium), environmental conditions (Temperature, Humidity, Rainfall), and pH to get a tailored crop recommendation.

- **Crop Recommendation**: The system uses the selected model to recommend the best crop based on the input values.

## Models Used

1. **Logistic Regression**: A simple yet effective model, which is good for linearly separable data.
2. **Decision Tree**: A tree-based algorithm that works well with complex datasets by splitting the data at various decision points.
3. **Support Vector Machine (SVM)**: A robust classifier that works well for high-dimensional spaces.
4. **Random Forest**: An ensemble learning method that combines multiple decision trees to enhance the model's accuracy.

## Dataset

The dataset used contains environmental factors and corresponding crop labels. The dataset is loaded using the `load_dataset()` function from the `app_functions.py` file. It consists of the following features:
- **Nitrogen (N)**: Nitrogen content in the soil.
- **Phosphorous (P)**: Phosphorous content in the soil.
- **Potassium (K)**: Potassium content in the soil.
- **Temperature**: The average temperature of the region.
- **Humidity**: The average humidity of the region.
- **pH**: The pH level of the soil.
- **Rainfall**: The average annual rainfall of the region.

The target variable is the crop label, which is mapped to a numeric value for training.

## Installation

### Prerequisites

Ensure you have Python installed. To install the required libraries, run:

```bash
pip install -r requirements.txt
```

### Required Libraries
- **numpy**
- **pandas**
- **seaborn**
- **scikit-learn**
- **matplotlib**
- **plotly**
- **joblib**

## Running the Application

1. **Clone the repository**:


2. **Install the dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the application**:
   ```bash
   streamlit run app.py
   ```

4. **Access the application**: After running the app, Streamlit will provide a local URL. Open it in your browser to access the interface.

## Training the Models

The models are trained using the data provided in `app_functions.py`. To train and save the models:
- The `save_models()` function fits and saves the four different models as `.pkl` files:
  - `logistic_regression_model.pkl`
  - `decision_tree_model.pkl`
  - `support_vector_model.pkl`
  - `random_forest_model.pkl`

You can call this function from within `app_functions.py` after splitting your data.

## Using the Application

1. **Model Selection**: Use the sidebar to select one of the available models. The accuracy of each model is shown alongside the name (e.g., Logistic Regression (0.987)).
  
2. **Input Parameters**: In the main interface, input the values for Nitrogen, Phosphorous, Potassium, temperature, humidity, pH, and rainfall.
  
3. **Predict**: Click the "Recommend Crop" button, and the application will display the best crop to cultivate based on your inputs.

4. **Error Handling**: If invalid inputs are provided, an error message will be displayed.

## Example

### User Input:

- Nitrogen (N): 39
- Phosphorous (P): 58
- Potassium (K): 85
- Temperature: 17.88
- Humidity: 15.40
- pH: 5.996
- Rainfall: 68.54

### Predicted Output:
```
Apple is the best crop to be cultivated.
```



## Contributors

- **Iranzi Innocent**: Developed the crop recommendation system for helping farmers make informed decisions about which crops to cultivate based on environmental factors.
