import streamlit as st
import pandas as pd
import requests

st.title('Loan Approval Prediction')

st.write('Enter the following information to predict loan approval:')

# Input fields
person_age = st.number_input('Age', min_value=18, max_value=100)
person_income = st.number_input('Annual Income', min_value=0)
person_emp_length = st.number_input('Employment Length (years)', min_value=0)
loan_amnt = st.number_input('Loan Amount', min_value=0)
loan_int_rate = st.number_input('Loan Interest Rate', min_value=0.0, max_value=100.0)
loan_percent_income = st.number_input('Loan Percent Income', min_value=0.0, max_value=100.0)
cb_person_cred_hist_length = st.number_input('Credit History Length (years)', min_value=0)

# Categorical inputs
person_home_ownership = st.selectbox('Home Ownership', ['RENT', 'OWN', 'MORTGAGE', 'OTHER'])
loan_intent = st.selectbox('Loan Intent', ['PERSONAL', 'EDUCATION', 'MEDICAL', 'VENTURE', 'HOMEIMPROVEMENT', 'DEBTCONSOLIDATION'])
loan_grade = st.selectbox('Loan Grade', ['A', 'B', 'C', 'D', 'E', 'F', 'G'])
cb_person_default_on_file = st.selectbox('Default on File', ['Y', 'N'])

if st.button('Predict Loan Approval'):
    # Prepare the input data
    input_data = {
        'person_age': person_age,
        'person_income': person_income,
        'person_emp_length': person_emp_length,
        'loan_amnt': loan_amnt,
        'loan_int_rate': loan_int_rate,
        'loan_percent_income': loan_percent_income,
        'cb_person_cred_hist_length': cb_person_cred_hist_length,
        'person_home_ownership': person_home_ownership,
        'loan_intent': loan_intent,
        'loan_grade': loan_grade,
        'cb_person_default_on_file': cb_person_default_on_file
    }

    # Send a POST request to the FastAPI endpoint
    response = requests.post('http://localhost:8000/predict', json=input_data)

    if response.status_code == 200:
        result = response.json()
        if result['loan_approval']:
            st.success('Loan Approved!')
        else:
            st.error('Loan Not Approved')
    else:
        st.error('Error in prediction. Please try again.')
