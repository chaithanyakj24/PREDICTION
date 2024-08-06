import pandas as pd
import streamlit as st
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import warnings
import os

# Ignore warnings
warnings.filterwarnings('ignore')

# Change background color to white
st.markdown(
    """
    <style>
    body {
        background-color: #FFFFFF;
        font-size: 16px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Add heading
st.title('PLACEMENT PREDICTOR')

# Read the dataset
data = pd.read_csv("new.csv")
data = data.drop(['S.NO'], axis=1)

# Map categorical variables to numerical values
data["GENDER"] = data["GENDER"].map({'Male': 0, 'Female': 1})
data["REGION"] = data["REGION"].map({' Rural': 0, ' Urban': 1})
data["SSLC_BOARD"] = data["SSLC_BOARD"].map({' State Board': 0, ' CBSE': 1, ' ICSE': 2})
data["PUC_BOARD"] = data["PUC_BOARD"].map({" State Board": 0, " CBSE": 1})
data["INTERNSHIP"] = data["INTERNSHIP"].map({" Yes": 0, " No": 1})
data["WORK EXPERIENCE"] = data["WORK EXPERIENCE"].map({" Yes": 0, " No": 1})
data["BRANCH"] = data["BRANCH"].map({" CS": 0, " ME": 1, " EE": 2, " IS": 3, " EC": 4, " CV": 5})
data["STATUS"] = data["STATUS"].map({"Placed": 0, "NotPlaced": 1})

# Define features (x) and target (y)
x = data.drop('STATUS', axis=1)
y = data['STATUS']

# Split the data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state=42)

# Train the Logistic Regression model
lr = LogisticRegression()
lr.fit(x_train, y_train)

# Predictions on the test set
y_pred = lr.predict(x_test)

# Calculate accuracy score
score1 = accuracy_score(y_test, y_pred)

# User Input
gender = st.number_input("Gender (0 for Male, 1 for Female):")
region = st.number_input("Region (0 for Rural, 1 for Urban):")
sslc_marks = st.number_input("SSLC Marks:")
sslc_board = st.number_input("SSLC Board (0 for State Board, 1 for CBSE, 2 for ICSE):")
puc_marks = st.number_input("PUC Marks:")
puc_board = st.number_input("PUC Board (0 for State Board, 1 for CBSE):")
cgpa = st.number_input("CGPA:")
internship = st.number_input("Internship (0 for Yes, 1 for No):")
work_experience = st.number_input("Work Experience (0 for Yes, 1 for No):")
branch = st.number_input("Branch (0 for CS, 1 for ME, 2 for EE, 3 for IS, 4 for EC, 5 for CV):")

# Add a predict button
if st.button("Predict"):
    test_data = pd.DataFrame({
        'GENDER': [gender],
        'REGION': [region],
        'SSLC_MARKS': [sslc_marks],
        'SSLC_BOARD': [sslc_board],
        'PUC_MARKS': [puc_marks],
        'PUC_BOARD': [puc_board],
        'CGPA': [cgpa],
        'INTERNSHIP': [internship],
        'WORK EXPERIENCE': [work_experience],
        'BRANCH': [branch],
    })

    # Make predictions on the test data point
    p = lr.predict(test_data)
    prob = lr.predict_proba(test_data)

    # Interpret the predictions
    if p == 1:
        st.error("Sorry for this time! Work harder and visit again.")
    else:
        st.success("Congratulations! You have the chances of getting placed.")

    # Store the inserted values in a CSV file
    if not os.path.exists('user_inputs.csv'):
        with open('user_inputs.csv', 'w') as f:
            test_data.to_csv(f, header=True, index=False)
    else:
        with open('user_inputs.csv', 'a') as f:
            test_data.to_csv(f, header=False, index=False)
            