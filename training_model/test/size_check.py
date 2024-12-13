import numpy as np
import pandas as pd
import tensorflow as tf
import os
import time

# Load CSV data
csv_file = "../component/realistic_motor_health_dataset.csv"  # Replace with your CSV file name
data = pd.read_csv(csv_file)

# Columns
input_columns = ["accel_x", "accel_y", "accel_z", "temperature", "cos_phi"]
label_column = "class"

X_test = data[input_columns].to_numpy().astype(np.float32)
y_test = data[label_column].to_numpy()

# Load models
h5_model = tf.keras.models.load_model("../src/motor_health_model.h5")
interpreter = tf.lite.Interpreter(model_path="motor_health_model_quantized_int8.tflite")
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

input_scale, input_zero_point = input_details[0]['quantization']
output_scale, output_zero_point = output_details[0]['quantization']

# Measure .h5 model latency
start_time_h5 = time.time()
h5_predictions = h5_model.predict(X_test)
end_time_h5 = time.time()
h5_latency = (end_time_h5 - start_time_h5) / len(X_test)
print(f"Average latency per sample for .h5 model: {h5_latency * 1000:.2f} ms")

# Measure .tflite model latency
start_time_tflite = time.time()
tflite_predictions = []
for i in range(X_test.shape[0]):
    input_data = X_test[i:i+1]
    quantized_input = np.round(input_data / input_scale + input_zero_point).astype(input_details[0]['dtype']) if input_scale > 0 else input_data
    interpreter.set_tensor(input_details[0]['index'], quantized_input)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])
    dequantized_output = output_scale * (output_data - output_zero_point) if output_scale > 0 else output_data
    tflite_predictions.append(dequantized_output)
end_time_tflite = time.time()
tflite_latency = (end_time_tflite - start_time_tflite) / len(X_test)
print(f"Average latency per sample for .tflite model: {tflite_latency * 1000:.2f} ms")

# Model sizes
h5_model_size = os.path.getsize("../src/motor_health_model.h5")
tflite_model_size = os.path.getsize("motor_health_model_quantized_int8.tflite")
print(f"Size of .h5 model: {h5_model_size / 1024:.2f} KB")
print(f"Size of .tflite model: {tflite_model_size / 1024:.2f} KB")
