import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


st.header("LinkedIn User Prediction Model")
st.write("Jeremy Brown")

income = st.selectbox("What is your household income?", 
                      options= ["Less than $10,000",
                                "10 to under $20,000",
                                "20 to under $30,000",
                                "30 to under $40,000",
                                "40 to under $50,000",
                                "50 to under $75,000",
                                "75 to under $100,000",
                                "100 to under $150,000",
                                "$150,000 or more",])
if income == "Less than $10,000":
    income = 1
elif income == "10 to under $20,000":
    income = 2
elif income == "20 to under $30,000":
    income = 3
elif income == "30 to under $40,000":
    income = 4
elif income == "40 to under $50,000":
    income = 5
elif income == "50 to under $75,000":
    income = 6
elif income == "75 to under $100,000":
    income = 7
elif income == "100 to under $150,000":
    income = 8
elif income == "$150,000 or more":
    income = 9

education = st.selectbox("What is your education level?", 
                         options= ["Less than high school (Grades 1-8 or no formal schooling)",
                                    "High school incomplete (Grades 9-11 or Grade 12 with NO diploma)",
                                    "High school graduate (Grade 12 with diploma or GED certificate)",
                                    "Some college, no degree (includes some community college)",
                                    "Two-year associate degree from a college or university",
                                    "Four-year college or university degree/Bachelor’s degree (e.g., BS, BA, AB)",
                                    "Some postgraduate or professional schooling, no postgraduate degree (e.g. some graduate school)",
                                    "Postgraduate or professional degree, including master’s, doctorate, medical or law degree (e.g., MA, MS, PhD, MD, JD)",])
if education == "Less than high school (Grades 1-8 or no formal schooling)":
    education = 1
elif education == "High school incomplete (Grades 9-11 or Grade 12 with NO diploma)":
    education = 2
elif education == "High school graduate (Grade 12 with diploma or GED certificate)":
    education = 3
elif education == "Some college, no degree (includes some community college)":
    education = 4
elif education == "Two-year associate degree from a college or university":
    education = 5
elif education == "Four-year college or university degree/Bachelor’s degree (e.g., BS, BA, AB)":
    education = 6
elif education == "Some postgraduate or professional schooling, no postgraduate degree (e.g. some graduate school)":
    education = 7
elif education == "Postgraduate or professional degree, including master’s, doctorate, medical or law degree (e.g., MA, MS, PhD, MD, JD)":
    education = 8

parent = st.radio("Are you a parent of a child under 18 living in your home? ", ["Yes", "No"])
if parent == "Yes":
    parent = 1
else:
    parent = 2

marital_status = st.radio("Select marital status?", ["Married", "Not Married"])
if marital_status == "Married":
    marital_status = 1
else:
    marital_status = 0

sex = st.radio("Select Gender", ["Male", "Female"])
if sex == "Male":
    sex = 0
else:
    sex = 1

age = st.number_input("Enter Age (max of 97)", value=0)


## To dataframe
input_data = pd.DataFrame({
    'income': [income],
    'education': [education],
    'parent': [parent],
    'married': [marital_status],
    'female': [sex],
    'age': [age]
})


# Read in CSV file
s = pd.read_csv("social_media_usage.csv")

def clean_sm(x):
    x = np.where(x == 1, 1, 0)
    return x
  
s['web1h'] = pd.to_numeric(s['web1h'], errors='coerce')
s["income"] = pd.to_numeric(s["income"], errors='coerce')
s["educ2"] = pd.to_numeric(s["educ2"], errors='coerce')
s['age'] = pd.to_numeric(s['age'], errors='coerce')

ss = pd.DataFrame({
    "sm_li": clean_sm(s['web1h']),
    "income1": np.where(s['income'] > 9, np.nan, s['income']),
    "education": np.where(s['educ2'] > 8, np.nan, s['educ2']),
    "par1": clean_sm(s['par']),
    "marital1": clean_sm(s['marital']),
    "female": np.where(s['sex'] == 1, 0, 1),
    "age1": np.where(s['age'] > 98, np.nan, s['age'])
})

# Define the new column names and order
new_columns = {'income1': 'income', 'par1': 'parent', 'marital1': 'married', 'age1': 'age'}
new_order = ['income', 'education', 'parent', 'married', 'female','age', 'sm_li']

# Rename columns
ss.rename(columns=new_columns, inplace=True)

# Reorder columns and create dataframe 'ss'
ss = ss[new_order]
ss = ss.dropna()

# Create target vector (y) and feature set (x)
y = ss["sm_li"]
x = ss.drop("sm_li", axis=1)

# Split training and testing data withholding 20% for testing
x_train, x_test, y_train, y_test = train_test_split(x,
                                                    y,
                                                    stratify=y,
                                                    test_size=.2,
                                                    random_state=216
                                                )


# Instantiate a logistic regression model and set class_weight to balanced. Fit the model with the training data
lr = LogisticRegression(class_weight='balanced')
lr.fit(x_train, y_train)

# Using model to make predictions
y_pred = lr.predict(x_test)


if st.button("Submit"):
    if education:
        # Make predictions
        prediction = lr.predict(input_data)
        probabilities = lr.predict_proba(input_data)

        # Display results
        st.write("Predicted Class:", prediction)
        st.write("Predicted Probabilities:", probabilities)

    else:
        st.warning("Please enter input data.")




