#include <stdio.h>
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "esp_log.h"

// Tensorflow library
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"

// Model
#include "motor_health_model_quantized_int8.hpp"

// TensorFlow Lite Micro configuration
constexpr int kTensorArenaSize = 10 * 1024;  // Adjust size based on the model requirements
uint8_t tensor_arena[kTensorArenaSize];

extern "C" void app_main() {
    printf("Starting Motor Health Classification...\n");

    // Load the TensorFlow Lite model from the C array
    const tflite::Model* model = tflite::GetModel(motor_health_model_quantized_int8_tflite);
    if (model->version() != TFLITE_SCHEMA_VERSION) {
        printf("Error: Model schema version %ld does not match supported version %d.\n",
               model->version(), TFLITE_SCHEMA_VERSION);
        return;
    }

    // Create MicroMutableOpResolver and register required operations
    tflite::MicroMutableOpResolver<4> resolver;  // Adjust the number for total operations
    resolver.AddFullyConnected();  // Fully connected layers
    resolver.AddSoftmax();         // Softmax for classification
    resolver.AddQuantize();        // Quantize operation for integer quantization
    resolver.AddDequantize();      // Dequantize operation

    // Build the interpreter
    tflite::MicroInterpreter interpreter(model, resolver, tensor_arena, kTensorArenaSize, nullptr);

    // Allocate memory for the tensors
    if (interpreter.AllocateTensors() != kTfLiteOk) {
        printf("Error: AllocateTensors() failed.\n");
        return;
    }

    // Get pointers to input and output tensors
    TfLiteTensor* input = interpreter.input(0);
    TfLiteTensor* output = interpreter.output(0);

    // Fill the input tensor with example data
    input->data.f[0] = 0.7;  // accel_x (1g)
    input->data.f[1] = -0.2; // accel_y (1g)
    input->data.f[2] = 0.05; // accel_z (1g)
    input->data.f[3] = 60.0; // temperature (°C)
    input->data.f[4] = 0.9; // cos(phi) (power factor)

    // Run inference
    if (interpreter.Invoke() != kTfLiteOk) {
        printf("Error: Model invocation failed.\n");
        return;
    }

    // Output predicted class probabilities
    printf("Predicted motor health class probabilities:\n");
    for (int i = 0; i < output->dims->data[1]; i++) {
        printf("Class %d: %f\n", i, output->data.f[i]);
    }

    // Determine the predicted class
    float max_prob = output->data.f[0];
    int predicted_class = 0;
    for (int i = 1; i < output->dims->data[1]; i++) {
        if (output->data.f[i] > max_prob) {
            max_prob = output->data.f[i];
            predicted_class = i;
        }
    }

    printf("Predicted Motor Health Class: %d (Probability: %f)\n", predicted_class, max_prob);
}