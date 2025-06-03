import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
import os

# Make sure the model directory exists
os.makedirs("model", exist_ok=True)

# Build a basic CNN
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
    MaxPooling2D(2, 2),
    Flatten(),
    Dense(64, activation='relu'),
    Dense(1, activation='sigmoid')  # Binary classifier
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Dummy data for testing only
X_dummy = tf.random.normal([10, 224, 224, 3])
y_dummy = tf.constant([0, 1, 0, 1, 1, 0, 0, 1, 1, 0])

# Train briefly
model.fit(X_dummy, y_dummy, epochs=2)

# Save model
model.save("model/skin_model.h5")
