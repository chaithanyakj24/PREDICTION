from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle
import numpy as np
from flask import Flask, send_from_directory
import os

app = Flask(__name__, static_folder='./my-app/build')
CORS(app)  # Enable CORS for all routes and origins

# Load the trained Logistic Regression model
with open('model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

@app.route('/')
def serve():
    return send_from_directory(app.static_folder, 'index.html')


@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get data from the request's JSON payload
        data = request.get_json()

        # Get the 'features' key from the JSON payload
        features = data.get('features')

        # Convert the input data to a NumPy array
        features_array = np.array(features).reshape(1, -1)

        # Make predictions using the loaded model
        prediction = model.predict(features_array)

        # Return the prediction as JSON
        return jsonify({'prediction': int(prediction[0])})

    except Exception as e:
        return jsonify({'error': str(e)})

if _name_ == '_main_':
    app.run(debug=True)