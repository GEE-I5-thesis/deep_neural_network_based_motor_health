import tensorflow as tf
import numpy as np

# Load the model
model = tf.keras.models.load_model("../src/motor_health_model.h5")  # Replace with your model file name
# print("Model loaded successfully!")

sample_input = np.array([[0.2, -0.2, 0.05, 60, 0.9]])  # Shape (1, 5)
# print("Input shape:", sample_input.shape)
# Predict the class probabilities
predictions = model.predict(sample_input)

# Get the class with the highest probability
predicted_class = np.argmax(predictions, axis=1)

print("Predicted class probabilities:", predictions)
print("Predicted class:", predicted_class[0])


