import pickle
import pandas as pd
import numpy as np
import streamlit as st
from tensorflow.keras.models import load_model

model = load_model('models/model.keras')

# Load Encoder

with open('models/label_encoder.pkl','rb') as file:
    label_encoder = pickle.load(file)

with open('models/one_hot_encoder.pkl','rb') as file:
    onhot_encoder = pickle.load(file)
with open('models/ordinal_encoder.pkl','rb') as file:
    ordinal_encoder = pickle.load(file)
with open('models/StandardScaler.pkl','rb') as file:
    stdsc = pickle.load(file)

st.title('Car Price Prediction')

brand = st.selectbox(
    "Select Brand",
    ['Mercedes',  'Hyundai',     'Tata',      'BMW',      'Kia',     'Ford',
     'Audi',    'Honda',   'Toyota',   'Maruti']
)

year = st.number_input(
    "Year",
    min_value=1990,
    max_value=2026,
    value=2020
)

kms_driven = st.number_input(
    "Kms Driven",
    min_value=0,
    value=30000
)
fuel_type = st.selectbox(
    "Fuel Type",
    ["Petrol", "Diesel", "CNG","Electric"]
)
Transmission = st.selectbox(
    "Transmission Type",
    ["Manual","Automatic"]
)
owner = st.selectbox(
    "Owner",
    ["First", "Second", "Third", "Fourth"]
)

engine_cc =st.number_input(
    'Engine CC',
    min_value=0
)
milage_kmpl = st.number_input(
    'Milage Kmpl',
    min_value=0
)
seats = st.number_input(
    'Seats',
    min_value=0
)

brand_fuel_type = np.array([[brand,fuel_type]])
brand_fuel_encoded = onhot_encoder.transform(brand_fuel_type)
owner = np.array([[owner]])
owner_encoded = ordinal_encoder.transform(owner)
transmission = np.array([Transmission])
transmission_encoded = label_encoder.transform(transmission).reshape(1, -1)
year= 2026-year
numeric_data = np.array([[year,kms_driven]])
final_data = np.hstack([
numeric_data,
transmission_encoded,
owner_encoded,
np.array([[engine_cc]]),
np.array([[milage_kmpl]]),
np.array([[seats]]),
brand_fuel_encoded
])

final_data = stdsc.transform(final_data)
prediction = model.predict(final_data)
st.success(
        f"Estimated Selling Price: ₹{prediction[0][0]:.2f} Lakhs"
    )
