import streamlit as st
import pandas as pd
import h2o
from h2o.automl import H2OAutoML
import time

# Initialize H2O
h2o.init()

# Load trained H2O AutoML model
model_path = "models/best_model.zip/GLM_1_AutoML_1_20231214_194418.zip"
model = h2o.import_mojo(model_path)

# Streamlit app title
st.title('Sales Conversion Prediction App')

# Function to make predictions
def predict(data):
    h2o_data = h2o.H2OFrame(data)
    prediction = model.predict(h2o_data)
    return prediction.as_data_frame()['predict'][0]

# User inputs
st.sidebar.header('User Input Data')

# Input fields
input_data = {}
input_data['Impressions'] = st.sidebar.number_input('Impressions', min_value=0)
input_data['Clicks'] = st.sidebar.number_input('Clicks', min_value=0)
input_data['Spent'] = st.sidebar.number_input('Spent', min_value=0.0)
input_data['Total_Conversion'] = st.sidebar.number_input('Total Conversion', min_value=0)
input_data['CPC'] = st.sidebar.number_input('Cost Per Click (CPC)', min_value=0.0)

# Make predictions on user input
if st.button('Predict'):
    prediction = predict(pd.DataFrame([input_data]))
    st.write(f"Predicted Approved Conversion: **{round(prediction, 2)}**")

def redirect(url):
    st.write(f"Redirecting to Neptune.ai...")
    st.write(f"Please wait...")
    st.write("Please note that you’ll need to sign in or register with a Neptune.ai account to view the full details of my project.")
    time.sleep(3)  # 3-second delay for demonstration
    st.markdown(f'<a href="{url}" target="_blank">Click here to go to Neptune.ai</a>', unsafe_allow_html=True)

# Button for redirection
if st.button('See Model Metrics on Neptune.ai'):
    redirect("https://app.neptune.ai/Vishal-Kumar-S/Sales-Conversion-Optimisation-MLOps-Project")

# Footer with attribution
st.text('Made with Streamlit and Neptune.ai')
