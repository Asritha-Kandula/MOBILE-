import streamlit as st
import joblib
import numpy as np

st.title("Mobile Price Predictor")

model = joblib.load("mobile_price_model.pkl")

features = []
for i in range(20):
    features.append(st.number_input(f"Feature {i+1}", value=0.0))

if st.button("Predict"):
    data = np.array(features).reshape(1, -1)
    prediction = model.predict(data)
    st.success(f"Predicted Price Range: {int(prediction[0])}")
