import streamlit as st
import pandas as pd
import pickle

# Load the pre-trained model
with open('best_model.pkl', 'rb') as f:
    model = pickle.load(f)

# Function to make predictions
def predict_stock_price(input_data):
    prediction = model.predict(input_data)
    return prediction

# Streamlit UI
st.title('Stock Price Prediction')

# Columns
col1, col2 = st.columns(2)

# Take user input for prediction

with col1:
    prev_close = st.number_input('Previous Close Price', min_value=0.0, value=1000.0)
    open_price = st.number_input('Open Price', min_value=0.0, value=1000.0)

with col2:
    high_price = st.number_input('High Price', min_value=0.0, value=1000.0)
    low_price = st.number_input('Low Price', min_value=0.0, value=1000.0)

# Create a DataFrame for prediction
input_data = pd.DataFrame({
    'Prev Close': [prev_close],
    'Open': [open_price],
    'High': [high_price],
    'Low': [low_price]
})

# Make prediction when the user presses the button
if st.button('Predict'):
    prediction = predict_stock_price(input_data)
    st.write(f"Predicted target stock price for today: {prediction[0]:.2f}")
