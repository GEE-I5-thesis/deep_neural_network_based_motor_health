import tensorflow as tf
import numpy as np
import pandas as pd

# Step 1: Load your trained model
model = tf.keras.models.load_model('../src/motor_health_model.h5')  # Replace with your model file path

# Step 2: Load your dataset (replace with your actual dataset)
dataset = pd.read_csv("../component/realistic_motor_health_dataset.csv")  # Replace with your CSV file name
X = dataset.iloc[:, :-1].values  # First 5 columns as inputs

# Step 3: Define the representative dataset function for quantization
def representative_data_gen():
    # Use a small subset of your training data to represent activations
    for data in X[:100]:  # You can choose 100 or a small number of samples
        yield [np.array(data, dtype=np.float32).reshape(1, 5)]  # Ensure shape matches input

# Step 4: Convert the model to TensorsFlow Lite format with Full Integer Quantization
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.OPTIMIZE_FOR_LATENCY]
converter.representative_dataset = representative_data_gen
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]

tflite_model = converter.convert()

with open('motor_health_model_quantized_int8.tflite', 'wb') as f:
    f.write(tflite_model)

print("Quantized model saved as 'motor_health_model_quantized_int8.tflite'")

