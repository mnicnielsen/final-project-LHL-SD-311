import streamlit as st
import pickle
import numpy as np
import xgboost
import pandas as pd

with open('../data/simple_model.pkl', 'rb') as file:
    xg_reg = pickle.load(file)

# xg_reg = load_model()

# regressor_loaded = data["model"]
# le_country = data["le_country"]
# le_education = data["le_education"]

def show_predict_page():
    st.title("San Diego 311 Request Response Time Prediction")

    st.write("""Please enter the information below to predict the response time for the 
    311 request you would like to make to the City of San Diego.""")

in_park_binary = (0,1)

council_districts = (1,2,3,4,5,6,7,8,9)

in_park = st.selectbox("in_park", in_park_binary)
council_district = st.selectbox("Council District", council_districts)

st.write(in_park)
st.write(council_district)

# ok = st.button("Predict Response Time")

# if ok == True:
#     X_dict = {'in_park': in_park, 'council_district': council_district}
#     X = pd.DataFrame(X_dict)
    
#     y_pred = model.predict(X)

#     response_time = xg_reg.predict(X)
#     st.write(response_time[0])
