import os
import cv2
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split


'''MAKE SURE TO TRAIN THIS IN KAGGLE OR GOOGLE COLAB AND SAVE IN A .H5 FORMAT'''
# Define your labels and their corresponding numerical values
labels = {'glioma': 0, 'meningioma': 1, 'notumor': 2, 'pituitary': 3}

# Specify the paths to your training and testing datasets
train_data_dir = 'path_to_training_dataset'
test_data_dir = 'path_to_testing_dataset'

# Function to load and preprocess the dataset
def load_dataset(data_dir):
    X = []  # Training Dataset
    Y = []  # Training Labels

    for label, value in labels.items():
        folder_path = os.path.join(data_dir, label)
        for image_file in os.listdir(folder_path):
            image_path = os.path.join(folder_path, image_file)
            image = cv2.imread(image_path)
            image = cv2.resize(image, (150, 150))
            X.append(image)
            Y.append(value)

    X = np.array(X)
    Y = np.array(Y)

    return X, Y

# Load and preprocess the training and testing datasets
X_train, Y_train = load_dataset(train_data_dir)
X_test, Y_test = load_dataset(test_data_dir)

# Shuffle the data
X_train, Y_train = shuffle(X_train, Y_train, random_state=42)

# Split the data into training, validation, and testing sets
X_train, X_valid, Y_train, Y_valid = train_test_split(X_train, Y_train, test_size=0.1, random_state=42)

# Convert labels to one-hot encoding
Y_train = to_categorical(Y_train)
Y_valid = to_categorical(Y_valid)
Y_test = to_categorical(Y_test)

# Create a CNN model
model = Sequential()
model.add(Conv2D(16, kernel_size=(3, 3), activation='relu', input_shape=(150, 150, 3)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(Dense(4, activation='softmax'))

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Scaling the features
X_train = X_train.astype('float32') / 255
X_valid = X_valid.astype('float32') / 255
X_test = X_test.astype('float32') / 255

# Train the model
history = model.fit(X_train, Y_train, validation_data=(X_valid, Y_valid), batch_size=32, epochs=20, verbose=1)

# Evaluate the model on the test dataset
test_loss, test_accuracy = model.evaluate(X_test, Y_test)
print(f'Test Accuracy: {test_accuracy * 100:.2f}%')

# Save the model
model.save('brain_tumor_classifier.h5')
