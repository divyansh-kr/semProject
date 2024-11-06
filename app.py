

import streamlit as st
import pandas as pd
import pickle
from sklearn.ensemble import RandomForestRegressor



# Custom CSS for styling
st.markdown("""
    <style>
        /* Set background and text colors */
        body {
            background-color: #F0F2F6;
            color: #2E2E2E;
            font-family: 'Helvetica', sans-serif;
        }
        .stButton>button {
            background-color: #6200EE;
            color: white;
            border-radius: 8px;
            padding: 0.75em 2em;
            font-size: 1em;
            font-weight: bold;
            transition: 0.3s ease;
        }
        .stButton>button:hover {
            background-color: #3700B3;
            color: #ffffff;
        }
        .stCheckbox, .stTextInput, .stNumberInput {
            padding: 1em;
        }
        .header-style {
            font-size: 2.5em;
            font-weight: bold;
            color: #3700B3;
        }
        .subheader-style {
            font-size: 1.5em;
            font-weight: bold;
            color: #6200EE;
        }
    </style>
""", unsafe_allow_html=True)

# Load the dataset and prepare data
data = pd.read_csv("C:/Users/divya/DATASETS/AQIdata1.csv")
target_column = "AQI"  
features_to_keep = ['PM-2.5 conc', 'PM-10 conc', 'NO2 conc', 'SO2 conc', 'CO conc', 'Ozone conc']
X = data[features_to_keep]
y = data[target_column].dropna()
X = X.loc[y.index]

# Check if model exists, else train and save
try:
    with open("C:/Users/divya/DATASETS/aqi_rf_reg_model.pkl", "rb") as f:
        model = pickle.load(f)
        expected_features = model.feature_names_in_
except (FileNotFoundError, ValueError, pickle.UnpicklingError):
    st.write("Training new model...")
    model = RandomForestRegressor()
    model.fit(X, y)
    expected_features = X.columns
    with open("C:/Users/divya/DATASETS/aqi_rf_reg_model.pkl", "wb") as f:
        pickle.dump(model, f)

# Title with new style


st.markdown('<div class="header-style">Air Quality Index (AQI) Prediction App</div>', unsafe_allow_html=True)
st.write("Analyze AQI data and predict AQI levels based on input features.")

# Sidebar for input form
st.sidebar.header("Enter AQI Prediction Inputs")
pm25 = st.sidebar.number_input("PM-2.5 conc", min_value=0.0, max_value=1000.0, value=10.0)
pm10 = st.sidebar.number_input("PM-10 conc", min_value=0.0, max_value=1000.0, value=10.0)
no2 = st.sidebar.number_input("NO2 conc", min_value=0.0, max_value=1000.0, value=10.0)
so2 = st.sidebar.number_input("SO2 conc", min_value=0.0, max_value=1000.0, value=10.0)
co = st.sidebar.number_input("CO conc", min_value=0.0, max_value=10.0, value=0.5)
ozone = st.sidebar.number_input("Ozone conc", min_value=0.0, max_value=10.0, value=0.05)

# Data sample display
if st.checkbox("Show data sample"):
    st.markdown('<div class="subheader-style">Data Sample</div>', unsafe_allow_html=True)
    st.write(data[features_to_keep + [target_column]].head())

# Construct the input DataFrame
input_data = pd.DataFrame([[pm25, pm10, no2, so2, co, ozone]], columns=features_to_keep)

# Prediction button with enhanced style
st.markdown('<div class="subheader-style">Prediction</div>', unsafe_allow_html=True)
if st.button("Predict AQI"):
    prediction = model.predict(input_data)
    st.success(f"Predicted AQI: {prediction[0]:.2f}")
