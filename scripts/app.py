import flask
from flask import Flask, render_template, request, jsonify
import numpy as np
import pickle

app = Flask(__name__)
# Load your model
model = pickle.load(open('model.pkl','rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()  # get data sent from front end
    # Example: Use data to make predictions
    # prediction = model.predict(np.array([data['feature1'], data['feature2']]))
    prediction = np.random.rand()  # Replace with actual prediction logic
    return jsonify({'prediction': prediction})

if __name__ == "__main__":
    app.run(debug=True)
