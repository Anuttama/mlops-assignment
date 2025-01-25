from flask import Flask, request, jsonify
import pickle
import numpy as np
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)

# Load the trained model from the pickle file
with open('model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

# Define a route to make predictions
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get data from POST request
        data = request.get_json(force=True)

        # Convert the input data into a numpy array and scale it
        input_data = np.array(data['input']).reshape(1, -1)
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(input_data)

        # Make prediction using the loaded model
        prediction = model.predict(scaled_data)

        # Return the result as a JSON response
        return jsonify({'prediction': int(prediction[0])})

    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)
