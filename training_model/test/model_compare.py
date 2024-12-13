import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

# Load CSV data
csv_file = "../component/realistic_motor_health_dataset.csv"  # Replace with your CSV file name
data = pd.read_csv(csv_file)

# Assume the CSV has input columns and one label column (modify as per your CSV structure)
input_columns = ["accel_x", "accel_y", "accel_z", "temperature", "cos_phi"]  # Modify with actual column names
label_column = "class"  # Replace with your label column name

X_test = data[input_columns].to_numpy().astype(np.float32)  # Convert inputs to FLOAT32
y_test = data[label_column].to_numpy()

# Load the .h5 model
h5_model = tf.keras.models.load_model("../src/motor_health_model.h5")

# Load the .tflite model
interpreter = tf.lite.Interpreter(model_path="motor_health_model_quantized_int8.tflite")
interpreter.allocate_tensors()

# Get input and output details for .tflite model
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Quantization parameters
input_scale, input_zero_point = input_details[0]['quantization']
output_scale, output_zero_point = output_details[0]['quantization']

# Predictions with .h5 model
h5_predictions = h5_model.predict(X_test)

# Predictions with .tflite model
tflite_predictions = []
for i in range(X_test.shape[0]):
    # Quantize input if needed
    input_data = X_test[i:i+1]
    if input_scale > 0:
        quantized_input = np.round(input_data / input_scale + input_zero_point).astype(input_details[0]['dtype'])
    else:
        quantized_input = input_data

    interpreter.set_tensor(input_details[0]['index'], quantized_input)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])

    # Dequantize output if needed
    if output_scale > 0:
        dequantized_output = output_scale * (output_data - output_zero_point)
    else:
        dequantized_output = output_data

    tflite_predictions.append(dequantized_output)

# Convert tflite_predictions to a numpy array
tflite_predictions = np.array(tflite_predictions).squeeze()  # Remove unnecessary dimensions

# Convert predictions to class labels (index of max probability)
y_pred_h5 = np.argmax(h5_predictions, axis=1)  # Shape should now be (5000,)
y_pred_tflite = np.argmax(tflite_predictions, axis=1)  # Shape should now be (5000,)

# Ensure both predictions are one-dimensional arrays
print("y_pred_h5 shape:", y_pred_h5.shape)
print("y_pred_tflite shape:", y_pred_tflite.shape)
print("y_test shape:", y_test.shape)

# Assert that all shapes are the same
assert y_pred_h5.shape == y_test.shape, f"Shape mismatch: {y_pred_h5.shape} vs {y_test.shape}"
assert y_pred_tflite.shape == y_test.shape, f"Shape mismatch: {y_pred_tflite.shape} vs {y_test.shape}"

# Calculate cumulative accuracy for each data point
accuracy_h5 = np.cumsum(y_pred_h5 == y_test) / np.arange(1, len(y_test) + 1)
accuracy_tflite = np.cumsum(y_pred_tflite == y_test) / np.arange(1, len(y_test) + 1)
overall_accuracy_h5 = np.mean(y_pred_h5 == y_test)
overall_accuracy_tflite = np.mean(y_pred_tflite == y_test)

# Print overall accuracy
print(f"Overall accuracy of .h5 model: {overall_accuracy_h5:.4f}")
print(f"Overall accuracy of .tflite model: {overall_accuracy_tflite:.4f}")

# Plot the curve graph
plt.plot(range(len(accuracy_h5)), accuracy_h5, label=".h5 Model Accuracy", color="blue", linewidth=2)
plt.plot(range(len(accuracy_tflite)), accuracy_tflite, label=".tflite Model Accuracy", color="green", linewidth=2)
plt.xlabel("Test Data Point Index")
plt.ylabel("Cumulative Accuracy")
plt.title("Cumulative Accuracy Curve for .h5 and .tflite Models")
plt.legend()
plt.grid(True)
plt.show()
