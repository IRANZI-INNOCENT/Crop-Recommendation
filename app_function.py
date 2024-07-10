# app_functions.py

import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
import joblib

# Load dataset and preprocess as needed
def load_dataset(file_path):
    import pandas as pd
    crop = pd.read_csv(file_path)

    # Map crop labels to numeric values
    crop_dict = {
        'apple': 1,
        'muskmelon': 2,
        'watermelon': 3,
        'grapes': 4,
        'pomegranate': 5,
        'lentil': 6,
        'blackgram': 7,
        'mungbean': 8,
        'mothbeans': 9,
        'pigeonpeas': 10,
        'kidneybeans': 11,
        'chickpea': 12,
    }
    crop['crop_num'] = crop['label'].map(crop_dict)

    # Split features and target variable
    X = crop.drop(['crop_num', 'label'], axis=1)
    y = crop['crop_num']

    return X, y

# Function to save models
def save_models(X_train, y_train):
    input_lr = [('polynomial', PolynomialFeatures(degree=2)), ('scale', StandardScaler()), ('model', LogisticRegression())]
    pipe_lr = Pipeline(input_lr)
    pipe_lr.fit(X_train, y_train)
    joblib.dump(pipe_lr, 'logistic_regression_model.pkl')

    input_tree = [('polynomial', PolynomialFeatures(degree=2)), ('scale', StandardScaler()), ('model', DecisionTreeClassifier())]
    pipe_tree = Pipeline(input_tree)
    pipe_tree.fit(X_train, y_train)
    joblib.dump(pipe_tree, 'decision_tree_model.pkl')

    input_sv = [('polynomial', PolynomialFeatures(degree=2)), ('scale', StandardScaler()), ('model', SVC())]
    pipe_sv = Pipeline(input_sv)
    pipe_sv.fit(X_train, y_train)
    joblib.dump(pipe_sv, 'support_vector_model.pkl')

    input_fore = [('polynomial', PolynomialFeatures(degree=2)), ('scale', StandardScaler()), ('model', RandomForestClassifier())]
    pipe_fore = Pipeline(input_fore)
    pipe_fore.fit(X_train, y_train)
    joblib.dump(pipe_fore, 'random_forest_model.pkl')

# Function to load models
def load_models(model_name):
    if model_name == 'logistic_regression':
        return joblib.load('A:\\APPS\\recommendation\\logistic_regression_model.pkl')
    elif model_name == 'decision_tree':
        return joblib.load('A:\\APPS\\recommendation\\decision_tree_model.pkl')
    elif model_name == 'support_vector':
        return joblib.load('A:\\APPS\\recommendation\\support_vector_model.pkl')
    elif model_name == 'random_forest':
        return joblib.load('A:\\APPS\\recommendation\\random_forest_model.pkl')
    else:
        raise ValueError("Invalid model choice. Please choose from 'logistic_regression', 'decision_tree', 'support_vector', 'random_forest'.")

# Function to recommend crop based on chosen model
def recommendation(model_pipe, N, P, k, temperature, humidity, ph, rainfall):
    features = np.array([[N, P, k, temperature, humidity, ph, rainfall]])
    prediction = model_pipe.predict(features)
    return prediction[0]


