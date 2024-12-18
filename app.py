from flask import Flask, request, render_template, jsonify
import joblib
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image
import io
import base64
import json

app = Flask(__name__)
linear_model = joblib.load('models/linear_regression_model.pkl')
kmeans_model = joblib.load('models/kmeans_model.pkl')
kmeans_scaler = joblib.load('models/kmeans_scaler.pkl')
neural_model = load_model('models/neural_network_model.h5')

@app.route('/')
def home():
    return render_template('index.html')
@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.form
        model_name = data.get('model')

        if model_name == 'linear_regression':
            input_data = json.loads(data.get('input'))
            input_array = np.array(input_data).reshape(1, -1)
            prediction = linear_model.predict(input_array)[0]
            return jsonify({'prediction': prediction})

        elif model_name == 'kmeans':
            input_data = json.loads(data.get('input'))
            input_array = np.array(input_data).reshape(1, -1)
            input_scaled = kmeans_scaler.transform(input_array)
            cluster = kmeans_model.predict(input_scaled)[0]
            return jsonify({'cluster': int(cluster)})

        elif model_name == 'neural_network':
            image_data = data.get('image')
            image = Image.open(io.BytesIO(base64.b64decode(image_data.split(',')[1])))
            image = image.convert('L').resize((28, 28))
            image_array = np.array(image) / 255.0
            image_array = image_array.reshape(1, 28, 28, 1)
            prediction = neural_model.predict(image_array)
            predicted_class = np.argmax(prediction, axis=1)[0]
            return jsonify({'prediction': int(predicted_class)})
        else:
            return jsonify({'error': 'Invalid model selected.'})
    except Exception as e:
        return jsonify({'error': str(e)})
if __name__ == '__main__':
    app.run(debug=True)
