import streamlit as st
import numpy as np
from app_function import load_dataset, save_models, load_models, recommendation

# Example input values
default_values = {
    'N': 39,
    'P': 58,
    'k': 85,
    'temperature': 17.88,
    'humidity': 15.40,
    'ph': 5.996,
    'rainfall': 68.54
}

# Load dataset
#X, y = load_dataset("A:\\ALIGORITHMS TRACKER\\REGRESSION\\Croprec_data.csv")

# Split data into training and testing sets
#from sklearn.model_selection import train_test_split
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Top bar for the system title
st.markdown("<h1 style='text-align: center; color: #4CAF50;'>Crop Recommendation</h1>", unsafe_allow_html=True)

# Display the sidebar with model selection
st.sidebar.title('Model Selection')
st.sidebar.subheader('Choose a Model')
model_choice = st.sidebar.selectbox('Select Model', ['Logistic Regression(0.987)', 'Decision Tree(0.991)', 'Support Vector Machine(0.975)', 'Random Forest(0.986)'])

# Display the parameters for user input in columns
st.header('Input Parameters')

col1, col2, col3 = st.columns(3)
with col1:
    N = st.number_input('Nitrogen', value=default_values['N'])
with col2:
    P = st.number_input('Phosphorous', value=default_values['P'])
with col3:
    k = st.number_input('Potassium', value=default_values['k'])

col4, col5, col6 = st.columns(3)
with col4:
    temperature = st.number_input('Temperature', value=default_values['temperature'])
with col5:
    humidity = st.number_input('Humidity', value=default_values['humidity'])
with col6:
    ph = st.number_input('pH', value=default_values['ph'])

rainfall = st.number_input('Rainfall', value=default_values['rainfall'])

# Perform recommendation on button click
if st.button('Recommend Crop'):
    try:
        model_name_mapping = {
            'Logistic Regression(0.987)': 'logistic_regression',
            'Decision Tree(0.991)': 'decision_tree',
            'Support Vector Machine(0.975)': 'support_vector',
            'Random Forest(0.986)': 'random_forest'
        }
        model_name = model_name_mapping[model_choice]
        model_pipe = load_models(model_name)
        predicted_crop = recommendation(model_pipe, N, P, k, temperature, humidity, ph, rainfall)

        crop_dict_rev = {
            1: "Apple", 2: "Muskmelon", 3: "Watermelon", 4: "Grapes", 5: "Pomegranate", 6: "Lentil", 7: "Blackgram",
            8: "Mungbean", 9: "Mothbeans", 10: "Pigeonpeas", 11: "Kidneybeans", 12: "Chickpea"
        }
        if predicted_crop in crop_dict_rev:
            st.success(f"**{crop_dict_rev[predicted_crop]}** is the best crop to be cultivated.")
        else:
            st.warning("Sorry, we are not able to recommend a proper crop for this environment.")
    except ValueError as e:
        st.error(str(e))

st.text('')
st.markdown('<p style="font-size:20px;">Developed By: AGRISAFE</p>', unsafe_allow_html=True)
