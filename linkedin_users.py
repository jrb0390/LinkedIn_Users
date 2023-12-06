# Import packages
import pandas as pd 
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Read in CSV file
s = pd.read_csv("C:\\Users\\casbrown\\Documents\\Project Prompt\\social_media_usage.csv")

# Check dimension of dataframe
s.shape

# Define function using np.where
def clean_sm(x):
    x = np.where(x == 1, 1, 0)
    return x

# Create a mock DataFrame with random integers
mock_data = np.random.randint(0, 4, size=(3, 2))

# Use mock data in clean_sm function
function_test = clean_sm(mock_data)

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

# Find sum of NaN values across all features
ss.isna().sum()

# Check data types
ss.dtypes

# Descriptive statistics
ss.describe()

# Pairplot using seaborn
sns.pairplot(ss, hue='sm_li', markers=["o", "s"], palette="husl")

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

# Evaluate Performance
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.2f}')

# Generate a confusion matrix
confusion_matrix(y_test, y_pred)

# Put confusion matrix into dataframe and label columns and index
pd.DataFrame(confusion_matrix(y_test, y_pred),
            columns=['Predicted negative','Predicted positive'],
            index=['Actual Negative', 'Actual Positive']).style.background_gradient(cmap='PiYG') 

# Evaluate Performance
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.2f}')

# Classification report
print('Classification Report:')
print(classification_report(y_test, y_pred))

# income=8, education=7, non-parent=0, married=1, female=1, age=42
new_person = pd.DataFrame({
    'income':[8],
    'education':[7],
    'parent': [0],
    'married':[1],
    'female':[1],
    'age': [42]
})

# Model run
new_person['sm_li'] = lr.predict(new_person) 

# income=8, education=7, non-parent=0, married=1, female=1, age=82
new_person_2 = pd.DataFrame({
    'income':[8],
    'education':[7],
    'parent': [0],
    'married':[1],
    'female':[1],
    'age': [82]
})

# Model run
new_person_2['sm_li'] = lr.predict(new_person_2) 