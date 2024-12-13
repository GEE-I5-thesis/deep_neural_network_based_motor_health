import numpy as np
import tensorflow as tf

# Load the TensorFlow Lite model
interpreter = tf.lite.Interpreter(model_path="motor_health_model_quantized_int8.tflite")
interpreter.allocate_tensors()

# Get input and output details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Check input tensor details
input_shape = input_details[0]['shape']
input_dtype = input_details[0]['dtype']
input_scale, input_zero_point = input_details[0]['quantization']  # Quantization params

# Input data (example data matching ESP32 input)
sample_input = np.array([[0.7, -0.2, 0.05, 60.0, 0.9]], dtype=np.float32)

# Quantize the input to match the model's requirements
if input_scale > 0:  # Only quantize if the model is quantized
    quantized_input = np.round(sample_input / input_scale + input_zero_point).astype(input_dtype)
else:
    quantized_input = sample_input

# Set the input tensor
interpreter.set_tensor(input_details[0]['index'], quantized_input)

# Run inference
interpreter.invoke()

# Get the output tensor
output_data = interpreter.get_tensor(output_details[0]['index'])

# Dequantize the output if necessary
output_scale, output_zero_point = output_details[0]['quantization']
if output_scale > 0:  # Only dequantize if the output is quantized
    dequantized_output = output_scale * (output_data - output_zero_point)
else:
    dequantized_output = output_data

# Display predicted class probabilities
print("Predicted motor health class probabilities:")
for i, prob in enumerate(dequantized_output[0]):  # Output is batch size x classes
    print(f"Class {i}: {prob}")

# Determine the predicted class
predicted_class = np.argmax(dequantized_output[0])
max_prob = dequantized_output[0][predicted_class]

print(f"Predicted Motor Health Class: {predicted_class} (Probability: {max_prob})")
