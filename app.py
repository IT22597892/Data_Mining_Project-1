import streamlit as st
import joblib
import pandas as pd

# Load the trained model
with open('model.joblib', 'rb') as model_file:
    model_pipeline = joblib.load(model_file)

# Title of the app
st.title("Term Deposit Subscription Prediction")

# User inputs
balance = st.number_input("Balance:", min_value=0.0, step=0.1)
day = st.number_input("Day of the month:", min_value=1, max_value=31, step=1)
campaign = st.number_input("Number of contacts performed during this campaign:", min_value=1, step=1)
duration = st.number_input("Duration of last contact (in seconds):", min_value=0, step=1)

# Categorical inputs
job = st.selectbox("Job type:", ['admin.', 'blue-collar', 'entrepreneur', 'housemaid', 'management',
                                  'retired', 'self-employed', 'student', 'technician', 'unemployed'])
marital = st.selectbox("Marital status:", ['divorced', 'married', 'single'])
education = st.selectbox("Education level:", ['primary', 'secondary', 'tertiary'])
default = st.selectbox("Credit in default?", ['yes', 'no'])
housing = st.selectbox("Has a housing loan?", ['yes', 'no'])
loan = st.selectbox("Has a personal loan?", ['yes', 'no'])
contact = st.selectbox("Contact communication type:", ['cellular', 'telephone'])
month = st.selectbox("Last contact month of the year:", ['jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec'])
poutcome = st.selectbox("Outcome of the previous marketing campaign:", ['failure', 'nonexistent', 'success'])

# Button to make predictions
if st.button("Predict"):
    # Prepare input data
    input_data = {
        'balance': balance,
        'day': day,
        'campaign': campaign,
        'duration': duration,
        'job': job,
        'marital': marital,
        'education': education,
        'default': default,
        'housing': housing,
        'loan': loan,
        'contact': contact,
        'month': month,
        'poutcome': poutcome
    }

    # Convert input data to DataFrame
    input_df = pd.DataFrame([input_data])

    # Make prediction
    prediction = model_pipeline.predict(input_df)
    result = "Yes" if prediction[0] == 1 else "No"

    # Display result
    st.success(f"Will the client subscribe to a term deposit? {result}")
