import streamlit as st
import pickle
import numpy as np

# def load_model():
#     with open('saved_steps.pkl', 'rb') as file:
#         data = pickle.load(file)
#     return data

# data = load_model()

# regressor_loaded = data["model"]
# le_country = data["le_country"]
# le_education = data["le_education"]

def show_predict_page()
    st.title()"San Diego 311 Request Response Time Prediction"

    st.write("""Please enter the information below to predict the response time!""")
