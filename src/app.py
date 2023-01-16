import streamlit as st
import numpy as np
import pandas as pd
import xgboost
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold
import pickle
import gensim

with open('../data/simple_model_g.pkl', 'rb') as file:
    xg_reg = pickle.load(file)

# Title and Page Description
st.title("San Diego 311 Request Response Time Prediction")
st.write("""Please enter the information below to predict the response time for the 
311 request you would like to make to the City of San Diego.""")

# Create dictionary of zeroes which will be populated by responses in Streamlit. 
X_dict = {
    'day_0': [0],'day_1': [0],'day_2': [0],'day_3': [0],'day_4': [0],'day_5': [0],'day_6': [0],
    'is_weekend': [0], 
    'month_1': [0],'month_2': [0],'month_3': [0],'month_4': [0],'month_5': [0],'month_6': [0],
    'month_7': [0],'month_8': [0],'month_9': [0],'month_10': [0],'month_11': [0],'month_12': [0],
    'in_park': [0], 
    'cd_1.0': [0],'cd_2.0': [0],'cd_3.0': [0],
    'cd_4.0': [0],'cd_5.0': [0],'cd_6.0': [0],
    'cd_7.0': [0],'cd_8.0': [0],'cd_9.0': [0],
    'is_phone': [0],
    'word_count_SS': [0],
    'COVID-19': [0],
    'Dead Animal': [0],
    'Development Services - Code Enforcement': [0],
    'Encampment': [0],
    'Environmental Services Code Compliance': [0],
    'Graffiti': [0],
    'Graffiti - Code Enforcement': [0],
    'Illegal Dumping': [0],
    'Missed Collection': [0],
    'Other': [0],
    'Oversized Vehicle': [0],
    'Parking': [0],
    'Pavement Maintenance': [0],
    'Pothole': [0],
    'ROW Maintenance': [0],
    'Right-of-Way Code Enforcement': [0],
    'Shared Mobility Device': [0],
    'Sidewalk Repair Issue': [0],
    'Stormwater': [0],
    'Stormwater Code Enforcement': [0],
    'Street Flooded': [0],
    'Street Light Maintenance': [0],
    'Street Sweeping': [0],
    'Traffic Engineering': [0],
    'Traffic Sign Maintenance': [0],
    'Traffic Signal Issue': [0],
    'Traffic Signal Timing': [0],
    'Trash/Recycling Collection': [0],
    'Tree Maintenance': [0],
    'Waste on Private Property': [0],
    'Weed Cleanup': [0],
    'w2v_0': [0], 'w2v_1': [0], 'w2v_2': [0], 'w2v_3': [0], 'w2v_4': [0], 
    'w2v_5': [0], 'w2v_6': [0], 'w2v_7': [0], 'w2v_8': [0], 'w2v_9': [0],
    'w2v_10': [0],'w2v_11': [0],'w2v_12': [0],'w2v_13': [0],'w2v_14': [0],
    'w2v_15': [0],'w2v_16': [0],'w2v_17': [0],'w2v_18': [0],'w2v_19': [0],
    'w2v_20': [0],'w2v_21': [0],'w2v_22': [0],'w2v_23': [0],'w2v_24': [0],
    'w2v_25': [0],'w2v_26': [0],'w2v_27': [0],'w2v_28': [0],'w2v_29': [0]
    }

# Create input boxes and assign dictionary values.
date_selected = st.date_input("Please enter the date for your request.")
in_park = st.selectbox("Does the request involve a location in a park?", ('No','Yes'))
council_district = st.selectbox("In which Council District is the location of the request?", (1,2,3,4,5,6,7,8,9))
is_phone = st.selectbox("How are you planning to make this request?",('Get It Done Mobile App', 'Get It Done Website', '311 Telephone Call'))

req_list = ['COVID-19', 'Dead Animal','Development Services - Code Enforcement','Encampment','Environmental Services Code Compliance','Graffiti','Graffiti - Code Enforcement','Illegal Dumping','Missed Collection','Other','Oversized Vehicle','Parking','Pavement Maintenance','Pothole','ROW Maintenance','Right-of-Way Code Enforcement','Shared Mobility Device','Sidewalk Repair Issue','Stormwater','Stormwater Code Enforcement','Street Flooded','Street Light Maintenance','Street Sweeping','Traffic Engineering','Traffic Sign Maintenance','Traffic Signal Issue','Traffic Signal Timing','Trash/Recycling Collection','Tree Maintenance','Waste on Private Property','Weed Cleanup']
type = st.selectbox("Please select the nature of your request.", req_list)

text = st.text_input('Please briefly describe your request:')

# 1. DATE FEATURES
weekday = date_selected.weekday()
month = date_selected.month

    # 1A. ASSIGN WEEKDAY VALUE TO DICTIONARY
i = 0
while i < 7: 
    if weekday == i: 
        X_dict[str('day_' + str(i))] = [1]
    i += 1

    # 1B. ASSIGN IS_WEEKEND VALUE
if weekday >= 5:
    X_dict['is_weekend'] = [1]

    # 1C. ASSIGN IS_MONTH VALUE
i = 1
while i < 13: 
    if month == i: 
        X_dict[str('month_' + str(i))] = [1]
    i += 1

# 2. IN_PARK FEATURE - assign dictionary value
if in_park == 'Yes': 
    X_dict['in_park'] = [1]

# 3. COUNCIL DISTRICT FEATURE
i = 1
while i < 10: 
    if council_district == i: 
        X_dict[str('cd_' + str(i) + '.0')] = [1]
    i += 1

# 4. IS_PHONE FEATURE
if is_phone == '311 Telephone Call':
    X_dict['is_phone'] = [1]

# 5. TEXT FEATURE
#import text packages
import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')
nltk.download('omw-1.4')
from nltk.corpus import stopwords
nltk.download('wordnet')
from nltk.stem import WordNetLemmatizer
nltk.download('punkt')
from nltk.stem import PorterStemmer
import string
import re

#import scaler as pkl
with open('../data/count_scaler.pkl', 'rb') as file:
    count_scaler = pickle.load(file)

#set stopwords
ENGstopwords = stopwords.words('english')

#define cleaning function

def clean(text):
    
    # remove punctuation    
    text = "".join([char for char in text if char not in string.punctuation])

    # tokenize words
    tokens = text.split()

    # remove all stopwords
    tokens_no_stopwords = [word for word in tokens if word not in ENGstopwords]

    # lemmatize
    lemmatizer = WordNetLemmatizer()
    lemmatized_tokens_n = [lemmatizer.lemmatize(token) for token in tokens_no_stopwords]
    lemmatized_tokens_v = [lemmatizer.lemmatize(token, pos ="v") for token in lemmatized_tokens_n]
    lemmatized_tokens_a = [lemmatizer.lemmatize(token, pos ="a") for token in lemmatized_tokens_v]
    lemmatized_tokens_r = [lemmatizer.lemmatize(token, pos ="r") for token in lemmatized_tokens_a]
    lemmatized_tokens_s = [lemmatizer.lemmatize(token, pos ="s") for token in lemmatized_tokens_r]
        
    return lemmatized_tokens_s

# define function to create tokens WITH stopwords
def count_all_words(text):
    
    # remove punctuation    
    text = "".join([char for char in text if char not in string.punctuation])

    # tokenize words
    tokens = text.split()
        
    return len(tokens)

#apploy functions
tokens = clean(text.lower())
word_count = count_all_words(text.lower())

#standardize
#create df with scaled word counts (shape 1,1)
word_count_SS = count_scaler.transform(pd.DataFrame({'word_count': [word_count]}))[0][0]

#assign dictionary value
X_dict['word_count_SS'][0] = word_count_SS

# 6. Type Feature
#create index from which to search for request type
for key in X_dict.keys():
    if key == type:
        X_dict[key][0] = 1

# 7. w2v Feature
# Step 1: Create a list of unique tokens from the words that were provided
unique_tokens = list(set(tokens))

# Step 2: Load pickled w2v model
with open('../data/Model_CBoW.pkl', 'rb') as file:
    Model_CBoW = pickle.load(file)

# Step 3: Create vector from text
text_vec = []
i = 0
while i < len(unique_tokens):
    next_word_vec = Model_CBoW.wv[unique_tokens[i]]
    
    j = 0
    while j < 30:
        if i == 0:
            text_vec.append(0)
        text_vec[j] = text_vec[j] + next_word_vec[j]
        j += 1
    i+= 1


# Step 4: Write a function to append each element of vector. The trick here will be import the saved pickles.  
def unpickle_scale(elem_num):  
    with open('../data/SS_scaler_w2v_' + str(elem_num) + '.pkl', 'rb') as file:
        scaler = pickle.load(file)
    X_dict['w2v_' + str(elem_num)] = list(scaler.transform(np.array(text_vec[elem_num]).reshape(-1, 1))[0])

for i in list(range(30)):
    unpickle_scale(i)


ok = st.button("Predict Response Time")

if ok == True:

    if (len(tokens) == 0):
        st.write("""Unfortunately, our system did not understand your request. Please be more descriptive or leave the text field blank. 
        The more descriptive you are, the better our prediction will be about the response time for your request! """)
    else:
        X = pd.DataFrame(X_dict)
        response_time = xg_reg.predict(X)
        # st.write(max(0,round(response_time[0])))
        st.write(text_vec)

    
    


    

