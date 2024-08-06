import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import pickle

# Load your dataset and preprocess it
data = pd.read_csv("C:\\Users\\chait\\OneDrive\\Desktop\\MINI PROJECT\\MINIPROJECT\\PREDICTION\\dataset.csv")

# Convert categorical variables into numerical using one-hot encoding
data = pd.get_dummies(data, columns=['GENDER', 'REGION', 'SSLC_BOARD', 'PUC_BOARD', 'INTERNSHIP', 'WORK EXPERIENCE', 'BRANCH'])

# Drop unnecessary columns if any
data = data.drop(['S.NO'], axis=1)

# Split the data into features (X) and target (y)
X = data.drop('STATUS', axis=1)
y = data['STATUS']

# Split the data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

# Scale the input features
scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)

# Train your logistic regression model with adjusted parameters
lr = LogisticRegression(max_iter=1000)  # Increase the number of iterations
lr.fit(x_train_scaled, y_train)

# Serialize and save the trained model to a file named 'model.pkl'
with open('model.pkl', 'wb') as model_file:
    pickle.dump(lr, model_file)
