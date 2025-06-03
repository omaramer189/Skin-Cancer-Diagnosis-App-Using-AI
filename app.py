from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
from PIL import Image
import io
import os

app = Flask(__name__)

# Load the trained CNN model (make sure model file exists in 'model/' folder)
MODEL_PATH = 'model/skin_model.h5'
model = tf.keras.models.load_model(MODEL_PATH)

# Preprocessing function to convert image to model input format
def preprocess(image_bytes):
    image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    image = image.resize((224, 224))
    img_array = np.array(image) / 255.0  # Normalize pixel values
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

@app.route('/')
def index():
    return "Skin Cancer Diagnosis AI - Flask API is Running"

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400

    file = request.files['image']
    image_bytes = file.read()

    try:
        input_data = preprocess(image_bytes)
        prediction = model.predict(input_data)[0][0]
        result = 'Cancer Detected' if prediction > 0.5 else 'No Cancer'

        return jsonify({
            'prediction': float(prediction),
            'result': result
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
