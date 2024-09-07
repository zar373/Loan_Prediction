import streamlit as st
import joblib
import pandas as pd
import numpy as np



# Loading models
DecTre = joblib.load("DecTre.joblib")
LogReg = joblib.load("LogReg.joblib")
RandFor = joblib.load("RandFor_model.joblib")
xgb = joblib.load("xgb_model.joblib")

# Custom CSS to change background color
st.markdown(
    """
    <style>
    .stApp {
        background-color: #84CEEB;
    }
    .stButton {
        color: #8860D0;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Page Title and Slogan
st.title("Loan Eligibility Predictor 🏦💡")
st.write("**Check Your Loan Eligibility Instantly with Multiple Machine Learning Models** 🚀")

# Introduction
st.write("""
    Welcome to the **Loan Eligibility Predictor**! This app uses powerful machine learning models—**Decision Tree**, 
    **Random Forest**, **XGBoost**, and **Logistic Regression**—to evaluate your loan eligibility. 
    Enter your details and see if you qualify for a loan today! 💼💰
""")

# Inputs
loanID = st.number_input("Enter Loan ID: ", placeholder='Enter Loan ID')

option_dependents = st.selectbox(
    "Select Number of Dependents:",
    ['0', '1', '2', '3']
)
dependents = int(option_dependents)  # Convert to integer

option_gender = st.radio(
    "Select Gender:",
    ['Male', 'Female']
)
gender = option_gender

option_married = st.radio(
    "Select Marital Status:",
    ['Yes', 'No']
)
married = option_married

option_education = st.radio(
    "Select Education:",
    ['Graduate', 'Not Graduate']
)
education = option_education

option_credit = st.radio(
    "Select Credit History",
    [1, 0]
)
credit = int(option_credit)

option_selfemp = st.radio(
    "Are you self-employed?",
    ['Yes', 'No']
)
selfemp = option_selfemp

appincome = st.number_input("Enter Applicant Income: ", min_value=100, max_value=100000, placeholder='e.g. 123')
coappincome = st.number_input("Enter Co-Applicant Income: ", min_value=100, max_value=100000, placeholder='e.g. 123')
loanamu = st.number_input("Enter Loan Amount: ", min_value=100, max_value=100000, placeholder='e.g. 123')
logloanamu = np.log(loanamu)
loanamuterm = st.number_input("Enter Loan Amount Term: ", min_value=100, max_value=100000, placeholder='e.g. 123')
option_area = st.selectbox(
    "Property Area: ",
    ['Rural', 'Semiurban', 'Urban']
)
prop_area = option_area

totalincome = appincome + coappincome
logtotalincome = np.log(totalincome)
emi = loanamu / loanamuterm
balanceincome = totalincome - emi

# Display computed values
st.write(f"Total Income: {totalincome}")
st.write(f"EMI: {emi}")
st.write(f"Balance Income: {balanceincome}")

# Convert inputs into a DataFrame
data = pd.DataFrame(
    {
        "Loan_ID": [loanID],
        "Gender": [gender],
        "Married": [married],
        "Dependents": [dependents],
        "Education": [education],
        "Self_Employed": [selfemp],
        "Credit_History": [credit],
        "Property_Area": [prop_area],
        "logTotalIncome": [logtotalincome],
        "logLoanAmount": [logloanamu],
        "EMI": [emi],
        "Balance_Income": [balanceincome],
        "Total_Income": [totalincome],
    }
)

# st.write(data)

# Drop the 'Loan_ID' column
data = data.drop(['Loan_ID'], axis=1)

#Process the data
data = pd.get_dummies(data, columns=['Gender', 'Married', 'Education', 'Self_Employed', 'Property_Area', 'Dependents'])

# Add missing columns
expected_columns = ['Credit_History', 'logLoanAmount', 'Gender_Female','Gender_Male', 'Married_No', 'Married_Yes', 'Dependents_3',
'Dependents_0', 'Dependents_1', 'Dependents_2', 'Education_Graduate',
'Education_Not Graduate', 'Self_Employed_No', 'Self_Employed_Yes',
'Property_Area_Rural', 'Property_Area_Semiurban', 'Property_Area_Urban', 'Total_Income', 'logTotalIncome', 'EMI', 'Balance_Income']

for column in expected_columns:
    if column not in data.columns:
        # st.warning(f"Adding missing column: {column}")
        data[column] = 0

data = data[expected_columns]

# Display processed data
st.subheader("Processed Data:")
# st.write(data)

# Select model
option_model = st.selectbox(
    "Select Model:",
        ['Decision Tree', 'Logistic Regression', 'Random Forest', 'XGBoost']
        )

# Prediction and error handling
if st.button("Predict!"):
    try:

        if option_model == 'Decision Tree':
            prediction = DecTre.predict(data)
        elif option_model == 'Logistic Regression':
            prediction = LogReg.predict(data)
        elif option_model == 'Random Forest':
            prediction = RandFor.predict(data)
        elif option_model == 'XGBoost':
            prediction = xgb.predict(data)

        if prediction[0] == 1:
            st.success("**Congratulations! 🎉 You are eligible for the loan.**")
        else:
            st.error("**Sorry, you are not eligible for the loan.**")

    except Exception as e:st.error(f"Error during data processing or model prediction: {str(e)}")
