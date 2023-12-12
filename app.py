<<<<<<< HEAD
import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


st.header("LinkedIn User Prediction Model")
st.write("Jeremy Brown")

# Household Income input & options
income = st.text_input("**What is your household income? (1-9) see below:**", key="household_income")
income_options = {
    "1": "Less than $10,000",
    "2": "10 to under $20,000",
    "3": "20 to under $30,000",
    "4": "30 to under $40,000",
    "5": "40 to under $50,000",
    "6": "50 to under $75,000",
    "7": "75 to under $100,000",
    "8": "100 to under $150,000",
    "9": "$150,000 or more",
}
st.write("<small>Income Options:</small>", unsafe_allow_html=True)
for code, label in income_options.items():
    st.write(f"<small>{code}-{label}</small>", unsafe_allow_html=True)

# Education input & options
education = st.text_input("**What is the highest level of school/degree completed? (1-8) see below:**", key="education_input")
education_options = {
    "1": "Less than high school (Grades 1-8 or no formal schooling)",
    "2": "High school incomplete (Grades 9-11 or Grade 12 with NO diploma)",
    "3": "High school graduate (Grade 12 with diploma or GED certificate)",
    "4": "Some college, no degree (includes some community college)",
    "5": "Two-year associate degree from a college or university",
    "6": "Four-year college or university degree/Bachelor’s degree (e.g., BS, BA, AB)",
    "7": "Some postgraduate or professional schooling, no postgraduate degree (e.g. some graduate school)",
    "8": "Postgraduate or professional degree, including master’s, doctorate, medical or law degree (e.g., MA, MS, PhD, MD, JD)",
}
st.write("<small>Education Options:<\small>", unsafe_allow_html=True)
for code, label in education_options.items():
    st.write(f"<small>{code}-{label}</small>", unsafe_allow_html=True)

# Parent input & options
parent = st. text_input("**Are you a parent of a child under 18 living in your home? (1, 2, 98, or 99) see below:**", key="parent_input")
yes_no_options = {
    "1": "Yes",
    "2": "No",
    "98": "Don't Know",
    "99": "Refused",
}
st.write("<small>Yes/No Options:</small>", unsafe_allow_html=True)
for code, label in yes_no_options.items():
    st.write(f"<small>{code}-{label}</small>", unsafe_allow_html=True)

# Marital status input & options
marital_status = st. text_input("**Current marital status (1-6, or 8, 9) see below:**", key="marital_input")
marital_status_options = {
    "1": "Married",
    "2": "Living with a partner",
    "3": "Divorced",
    "4": "Separated",
    "5": "Widowed",
    "6": "Never been married",
    "8": "Don't Know",
    "9": "Refused",
}
st.write("<small>Marital Status Options:</small>", unsafe_allow_html=True)
for code, label in marital_status_options.items():
    st.write(f"<small>{code}-{label}</small>", unsafe_allow_html=True)

# Sex input & options
sex = st.text_input("**What is your sex? (1, 2, 98, or 99):**", key="sex_input")
gender_options = {
    "1": "Male",
    "2": "Female",
    "98": "Don't Know",
    "99": "Refused",
}
st.write("<small>Gender Options:</small>", unsafe_allow_html=True)
for code, label in gender_options.items():
    st.write(f"<small>{code}-{label}</small>", unsafe_allow_html=True)

# Age input & options
age = st.text_input("**What is your age? (enter a number between 0 - 96, or max 97):**", key="age_input")
age_options = {
    "97": "97+",
}
st.write("<small>Age Options:</small>", unsafe_allow_html=True)
for code, label in age_options.items():
    st.write(f"<small>{code}-{label}</small>", unsafe_allow_html=True)


## To dataframe
input_data = pd.DataFrame({
    'income': income,
    'education': education,
    'parent': parent,
    'married':marital_status,
    'female':sex,
    'age': age
})


# Read in CSV file
s = pd.read_csv("social_media_usage.csv")

def user_prediction(dataframe, input_data):
    # Create 'ss' dataframe with specified features
    ss = pd.DataFrame({
        "sm_li": clean_sm(dataframe['web1h']),
        "income1": np.where(dataframe['income'] > 9, np.nan, dataframe['income']),
        "education": np.where(dataframe['educ2'] > 8, np.nan, dataframe['educ2']),
        "par1": clean_sm(dataframe['par']),
        "marital1": clean_sm(dataframe['marital']),
        "female": np.where(dataframe['sex'] == 1, 0, 1),
        "age1": np.where(dataframe['age'] > 98, np.nan, dataframe['age'])
    })

    # Define the new column names and order
    new_columns = {'income1': 'income', 'par1': 'parent', 'marital1': 'married', 'age1': 'age'}
    new_order = ['income', 'education', 'parent', 'married', 'female','age', 'sm_li']

    # Rename columns
    ss.rename(columns=new_columns, inplace=True)

    # Reorder columns and create dataframe 'ss'
    ss = ss[new_order]
    ss = ss.dropna()

    # Cast object data type to int
    ss["income"].astype(int)
    ss["education"].astype(int)
    ss['age'] = pd.to_numeric(ss['age'], errors='coerce')

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

    # Model run
    input_data['sm_li'] = lr.predict(input_data) 

    return input_data

user_prediction(s, input_data)
=======
import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


st.header("LinkedIn User Prediction Model")
st.write("Jeremy Brown")

# Household Income input & options
income = st.text_input("**What is your household income? (1-9) see below:**", key="household_income")
income_options = {
    "1": "Less than $10,000",
    "2": "10 to under $20,000",
    "3": "20 to under $30,000",
    "4": "30 to under $40,000",
    "5": "40 to under $50,000",
    "6": "50 to under $75,000",
    "7": "75 to under $100,000",
    "8": "100 to under $150,000",
    "9": "$150,000 or more",
}
st.write("<small>Income Options:</small>", unsafe_allow_html=True)
for code, label in income_options.items():
    st.write(f"<small>{code}-{label}</small>", unsafe_allow_html=True)

# Education input & options
education = st.text_input("**What is the highest level of school/degree completed? (1-8) see below:**", key="education_input")
education_options = {
    "1": "Less than high school (Grades 1-8 or no formal schooling)",
    "2": "High school incomplete (Grades 9-11 or Grade 12 with NO diploma)",
    "3": "High school graduate (Grade 12 with diploma or GED certificate)",
    "4": "Some college, no degree (includes some community college)",
    "5": "Two-year associate degree from a college or university",
    "6": "Four-year college or university degree/Bachelor’s degree (e.g., BS, BA, AB)",
    "7": "Some postgraduate or professional schooling, no postgraduate degree (e.g. some graduate school)",
    "8": "Postgraduate or professional degree, including master’s, doctorate, medical or law degree (e.g., MA, MS, PhD, MD, JD)",
}
st.write("<small>Education Options:<\small>", unsafe_allow_html=True)
for code, label in education_options.items():
    st.write(f"<small>{code}-{label}</small>", unsafe_allow_html=True)

# Parent input & options
parent = st. text_input("**Are you a parent of a child under 18 living in your home? (1, 2, 98, or 99) see below:**", key="parent_input")
yes_no_options = {
    "1": "Yes",
    "2": "No",
    "98": "Don't Know",
    "99": "Refused",
}
st.write("<small>Yes/No Options:</small>", unsafe_allow_html=True)
for code, label in yes_no_options.items():
    st.write(f"<small>{code}-{label}</small>", unsafe_allow_html=True)

# Marital status input & options
marital_status = st. text_input("**Current marital status (1-6, or 8, 9) see below:**", key="marital_input")
marital_status_options = {
    "1": "Married",
    "2": "Living with a partner",
    "3": "Divorced",
    "4": "Separated",
    "5": "Widowed",
    "6": "Never been married",
    "8": "Don't Know",
    "9": "Refused",
}
st.write("<small>Marital Status Options:</small>", unsafe_allow_html=True)
for code, label in marital_status_options.items():
    st.write(f"<small>{code}-{label}</small>", unsafe_allow_html=True)

# Sex input & options
sex = st.text_input("**What is your sex? (1, 2, 98, or 99):**", key="sex_input")
gender_options = {
    "1": "Male",
    "2": "Female",
    "98": "Don't Know",
    "99": "Refused",
}
st.write("<small>Gender Options:</small>", unsafe_allow_html=True)
for code, label in gender_options.items():
    st.write(f"<small>{code}-{label}</small>", unsafe_allow_html=True)

# Age input & options
age = st.text_input("**What is your age? (enter a number between 0 - 96, or max 97):**", key="age_input")
age_options = {
    "97": "97+",
}
st.write("<small>Age Options:</small>", unsafe_allow_html=True)
for code, label in age_options.items():
    st.write(f"<small>{code}-{label}</small>", unsafe_allow_html=True)


## To dataframe
input_data = pd.DataFrame({
    'income': income,
    'education': education,
    'parent': parent,
    'married':marital_status,
    'female':sex,
    'age': age
})


# Read in CSV file


def user_prediction(input_data):
    import pandas as pd
    import numpy as np
    import seaborn as sns
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

    s = pd.read_csv("social_media_usage.csv")
    # Create 'ss' dataframe with specified features
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

    # Cast object data type to int
    ss["income"].astype(int)
    ss["education"].astype(int)
    ss['age'] = pd.to_numeric(ss['age'], errors='coerce')

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

    # Model run
    input_data['sm_li'] = lr.predict(input_data) 

    return input_data

user_prediction(s, input_data)
>>>>>>> 368e4fae2635a02e106429996b6f3939c40cb366






