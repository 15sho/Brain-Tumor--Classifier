import os
import cv2
import numpy as np
from tkinter import *
from tkinter import filedialog
from keras.models import load_model

# Define your labels and their corresponding numerical values
labels = {'glioma': 0, 'meningioma': 1, 'notumor': 2, 'pituitary': 3}

# Load the trained model
model = load_model('brain_tumor_classifier.h5')  # Replace with the path to your saved model

# Function to load and preprocess a single image
def load_and_preprocess_image(image_path):
    image = cv2.imread(image_path)
    image = cv2.resize(image, (150, 150))
    image = image.astype('float32') / 255
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image

# Function to predict the tumor type for a single image
def predict_tumor_type(image_path):
    image = load_and_preprocess_image(image_path)
    predictions = model.predict(image)
    predicted_class = labels[list(labels.keys())[np.argmax(predictions)]]
    return predicted_class

# Function to open a file dialog for image selection
def open_file_dialog():
    file_path = filedialog.askopenfilename()
    if file_path:
        predicted_tumor_type = predict_tumor_type(file_path)
        result_label.config(text="Predicted Tumor Type: " + predicted_tumor_type)

# Create a simple GUI
root = Tk()
root.title("Brain Tumor Classifier")

# Create a "Load Image" button
load_button = Button(root, text="Load Image", command=open_file_dialog)
load_button.pack()

# Create a label to display the result
result_label = Label(root, text="")
result_label.pack()

root.mainloop()
