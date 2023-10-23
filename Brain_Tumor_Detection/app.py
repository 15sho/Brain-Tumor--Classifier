from flask import Flask, render_template, request
import os
import cv2
import numpy as np
from keras.models import load_model

app = Flask(__name__)

# Define your labels and their corresponding numerical values
labels = {'glioma': 0, 'meningioma': 1, 'notumor': 2, 'pituitary': 3}

# Load the trained model
model = load_model('brain_tumor_classifier.h5')  # Replace with the path to your saved model

# Function to load and preprocess an uploaded image
def load_and_preprocess_image(image):
    image = cv2.imdecode(np.fromstring(image.read(), np.uint8), cv2.IMREAD_COLOR)
    image = cv2.resize(image, (150, 150))
    image = image.astype('float32') / 255
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image

# Function to predict the tumor type for an uploaded image
def predict_tumor_type(image):
    image = load_and_preprocess_image(image)
    predictions = model.predict(image)
    predicted_class = labels[list(labels.keys())[np.argmax(predictions)]]
    return predicted_class

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        uploaded_image = request.files['image']
        if uploaded_image:
            predicted_class = predict_tumor_type(uploaded_image)
            predicted_label = list(labels.keys())[predicted_class]
            return render_template('result.html', result=predicted_label)
    return render_template('index.html')


if __name__ == '__main__':
    app.run(debug=True)
