import streamlit as st
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
import pandas as pd
import pickle

# Load the trained model, scaler, and encoders
model = tf.keras.models.load_model('model.h5')

# Load the encoders and scaler
with open('label_encoder_gender.pkl', 'rb') as file:
    label_encoder_gender = pickle.load(file)

with open('one_hot_enc_geo.pkl', 'rb') as file:
    one_hot_enc_geo = pickle.load(file)  # Use the correct variable name

with open('Scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)

## Streamlit app
st.title('Bank Customer Churn Prediction')

# User input
geography = st.selectbox('Geography', one_hot_enc_geo.categories_[0])
gender = st.selectbox('Gender', label_encoder_gender.classes_)
age = st.slider('Age', 18, 92)
balance = st.number_input('Balance', value=0.0)
credit_score = st.number_input('Credit Score', value=0)
estimated_salary = st.number_input('Estimated Salary', value=0.0)
tenure = st.slider('Tenure', 0, 10)
num_of_products = st.slider('Number of Products', 1, 4)
has_cr_card = st.selectbox('Has Credit Card', [0, 1])
is_active_member = st.selectbox('Is Active Member', [0, 1])

# Prepare the input data
input_data = pd.DataFrame({
    'CreditScore': [credit_score],
    'Gender': [label_encoder_gender.transform([gender])[0]],
    'Age': [age],
    'Tenure': [tenure],
    'Balance': [balance],
    'NumOfProducts': [num_of_products],
    'HasCrCard': [has_cr_card],
    'IsActiveMember': [is_active_member],
    'EstimatedSalary': [estimated_salary]
})

# One-hot encode 'Geography'
geo_encoded = one_hot_enc_geo.transform([[geography]]).toarray()
geo_encoded_df = pd.DataFrame(geo_encoded, columns=one_hot_enc_geo.get_feature_names_out(['Geography']))

# Combine one-hot encoded columns with input data
input_data = pd.concat([input_data.reset_index(drop=True), geo_encoded_df], axis=1)

# Ensure correct feature order before scaling
expected_columns = scaler.feature_names_in_  # This contains the correct feature order
input_data = input_data.reindex(columns=expected_columns)

# Debugging checks
st.write("Final Input Data before scaling:")
st.write(input_data)

st.write("Missing values before scaling:", input_data.isna().sum())

# Scale the input data
input_data_scaled = scaler.transform(input_data)

st.write("Missing values after scaling:", pd.DataFrame(input_data_scaled).isna().sum())

# Predict churn
prediction = model.predict(input_data_scaled)
prediction_proba = prediction[0][0]

st.write(f'Churn Probability: {prediction_proba:.2f}')

if prediction_proba > 0.5:
    st.write('The customer is likely to churn.')
else:
    st.write('The customer is not likely to churn.')