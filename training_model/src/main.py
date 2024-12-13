import time  # Import the time module for measuring execution time
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt

# Step 1: Load the dataset from CSV
dataset = pd.read_csv("../component/realistic_motor_health_dataset.csv")  # Replace with your CSV file name

# Step 2: Separate features and labels
X = dataset.iloc[:, :-1].values  # First 5 columns as inputs
y = dataset.iloc[:, -1].values   # Last column as labels (class)

# Step 3: Define the model architecture
model = tf.keras.Sequential([
    tf.keras.layers.InputLayer(input_shape=(5,)),  # Input layer for 5 features
    tf.keras.layers.Dense(128, activation='relu'), # First hidden layer
    tf.keras.layers.Dense(64, activation='relu'),  # Second hidden layer
    tf.keras.layers.Dense(32, activation='relu'),  # Third hidden layer
    tf.keras.layers.Dense(4, activation='softmax') # Output layer for 4 classes
])

# Measure the time taken to compile the model
start_compile_time = time.time()
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)
end_compile_time = time.time()
print(f"Time taken to compile the model: {end_compile_time - start_compile_time:.4f} seconds")

# Step 5: Train-test split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Measure the time taken to train the model
start_training_time = time.time()
history = model.fit(X_train, y_train, epochs=20, batch_size=32, validation_data=(X_test, y_test))
end_training_time = time.time()
print(f"Time taken to train the model: {end_training_time - start_training_time:.4f} seconds")

# Step 7: Display final accuracies
final_train_accuracy = history.history['accuracy'][-1]
final_val_accuracy = history.history['val_accuracy'][-1]

print(f"Final Training Accuracy: {final_train_accuracy:.2f}")
print(f"Final Validation Accuracy: {final_val_accuracy:.2f}")

# Step 8: Save the model (uncomment if needed)
# model.save("motor_health_model.h5")
# print("Model saved as motor_health_model.h5")

# Optional: Plot training history
# plt.figure(figsize=(12, 4))
# # Plot training & validation accuracy values
# plt.subplot(1, 2, 1)
# plt.plot(history.history['accuracy'], label='Train Accuracy')
# plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
# plt.title('Model Accuracy')
# plt.xlabel('Epoch')
# plt.ylabel('Accuracy')
# plt.legend(loc='lower right')

# # Plot training & validation loss values
# plt.subplot(1, 2, 2)
# plt.plot(history.history['loss'], label='Train Loss')
# plt.plot(history.history['val_loss'], label='Validation Loss')
# plt.title('Model Loss')
# plt.xlabel('Epoch')
# plt.ylabel('Loss')
# plt.legend(loc='upper right')

# plt.tight_layout()
# plt.show()
