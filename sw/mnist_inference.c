// =============================================================================
// File: mnist_inference.c
// Description: MNIST Handwritten Digit Classification using NeuroRISC NPU
// 
// Network Architecture:
// - Input: 784 pixels (28×28 grayscale image)
// - Layer 1: Fully connected 784 → 128 with ReLU
// - Layer 2: Fully connected 128 → 10 with Softmax (argmax for prediction)
// 
// Quantization: 8-bit weights and activations (Q7 format)
// =============================================================================

#include "neurisc_runtime.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// =============================================================================
// Network Configuration
// =============================================================================

#define INPUT_SIZE      784  // 28×28 pixels
#define HIDDEN_SIZE     128  // Hidden layer neurons
#define OUTPUT_SIZE     10   // 10 digit classes (0-9)

// =============================================================================
// Network Parameters (would be loaded from trained model)
// =============================================================================

// Layer 1: 784 → 128
static q7_t layer1_weights[HIDDEN_SIZE * INPUT_SIZE];  // 128×784
static q31_t layer1_bias[HIDDEN_SIZE];                 // 128

// Layer 2: 128 → 10
static q7_t layer2_weights[OUTPUT_SIZE * HIDDEN_SIZE]; // 10×128
static q31_t layer2_bias[OUTPUT_SIZE];                 // 10

// =============================================================================
// Intermediate Buffers
// =============================================================================

static q7_t  input_buffer[INPUT_SIZE];       // Quantized input image
static q15_t hidden_output[HIDDEN_SIZE];     // Hidden layer output
static q15_t network_output[OUTPUT_SIZE];    // Final network output

// =============================================================================
// Performance Counters
// =============================================================================

typedef struct {
    uint32_t layer1_cycles;
    uint32_t layer2_cycles;
    uint32_t total_cycles;
} perf_stats_t;

static perf_stats_t perf_stats;

// =============================================================================
// Model Initialization (Demo weights - would load from file in practice)
// =============================================================================

void mnist_init_model(void) {
    printf("Initializing MNIST model...\n");
    
    // Initialize with small random values (demo only)
    // In practice, load pre-trained weights from flash/SD card
    
    for (int i = 0; i < HIDDEN_SIZE * INPUT_SIZE; i++) {
        // Simple pseudo-random initialization
        layer1_weights[i] = (q7_t)((i * 7 + 13) % 31 - 15);
    }
    
    for (int i = 0; i < HIDDEN_SIZE; i++) {
        layer1_bias[i] = 0;
    }
    
    for (int i = 0; i < OUTPUT_SIZE * HIDDEN_SIZE; i++) {
        layer2_weights[i] = (q7_t)((i * 11 + 7) % 31 - 15);
    }
    
    for (int i = 0; i < OUTPUT_SIZE; i++) {
        layer2_bias[i] = 0;
    }
    
    printf("Model initialized with %d parameters\n",
           HIDDEN_SIZE * INPUT_SIZE + HIDDEN_SIZE + 
           OUTPUT_SIZE * HIDDEN_SIZE + OUTPUT_SIZE);
}

// =============================================================================
// Image Preprocessing
// =============================================================================

void mnist_preprocess_image(const uint8_t* image, q7_t* output) {
    // Normalize and quantize image
    // Input: 8-bit grayscale (0-255)
    // Output: Q7 format (-128 to 127)
    
    for (int i = 0; i < INPUT_SIZE; i++) {
        // Normalize to [-1, 1] range and quantize
        float normalized = (image[i] / 255.0f) * 2.0f - 1.0f;
        output[i] = quantize_q7(normalized, 1.0f / 127.0f);
    }
}

// =============================================================================
// Softmax and Prediction
// =============================================================================

int mnist_argmax(const q15_t* output, int size) {
    int max_idx = 0;
    q15_t max_val = output[0];
    
    for (int i = 1; i < size; i++) {
        if (output[i] > max_val) {
            max_val = output[i];
            max_idx = i;
        }
    }
    
    return max_idx;
}

void mnist_softmax(const q15_t* input, float* output, int size) {
    // Convert Q15 to float for softmax computation
    float exp_sum = 0.0f;
    float max_val = dequantize_q15(input[0]);
    
    // Find max for numerical stability
    for (int i = 1; i < size; i++) {
        float val = dequantize_q15(input[i]);
        if (val > max_val) max_val = val;
    }
    
    // Compute exp and sum
    for (int i = 0; i < size; i++) {
        float val = dequantize_q15(input[i]) - max_val;
        // Simple exp approximation for embedded system
        float exp_val = (val > -5.0f) ? (1.0f + val + val*val/2.0f) : 0.001f;
        output[i] = exp_val;
        exp_sum += exp_val;
    }
    
    // Normalize
    for (int i = 0; i < size; i++) {
        output[i] /= exp_sum;
    }
}

// =============================================================================
// Forward Pass Through Network
// =============================================================================

int mnist_infer(const uint8_t* image, float* probabilities) {
    uint32_t start_cycles, end_cycles;
    
    // Initialize NPU
    npu_init();
    
    // Preprocess input image
    mnist_preprocess_image(image, input_buffer);
    
    printf("\n=== Starting MNIST Inference ===\n");
    
    // ------------------------------------------------------------------
    // Layer 1: Fully Connected 784 → 128 with ReLU
    // ------------------------------------------------------------------
    printf("Layer 1: FC(784 → 128) + ReLU\n");
    
    start_cycles = npu_get_performance_cycles();
    
    npu_fully_connected(
        layer1_weights,     // Weights: 128×784
        input_buffer,       // Input: 1×784
        layer1_bias,        // Bias: 128
        hidden_output,      // Output: 1×128
        INPUT_SIZE,         // Input size: 784
        HIDDEN_SIZE,        // Output size: 128
        ACTIVATION_RELU     // ReLU activation
    );
    
    end_cycles = npu_get_performance_cycles();
    perf_stats.layer1_cycles = end_cycles - start_cycles;
    printf("  Completed in %u cycles\n", perf_stats.layer1_cycles);
    
    // ------------------------------------------------------------------
    // Layer 2: Fully Connected 128 → 10 (Linear)
    // ------------------------------------------------------------------
    printf("Layer 2: FC(128 → 10) + Linear\n");
    
    start_cycles = npu_get_performance_cycles();
    
    npu_fully_connected(
        layer2_weights,     // Weights: 10×128
        (q7_t*)hidden_output, // Input: 1×128 (cast from Q15, assumes requantization)
        layer2_bias,        // Bias: 10
        network_output,     // Output: 1×10
        HIDDEN_SIZE,        // Input size: 128
        OUTPUT_SIZE,        // Output size: 10
        ACTIVATION_LINEAR   // Linear activation (softmax done in software)
    );
    
    end_cycles = npu_get_performance_cycles();
    perf_stats.layer2_cycles = end_cycles - start_cycles;
    printf("  Completed in %u cycles\n", perf_stats.layer2_cycles);
    
    perf_stats.total_cycles = perf_stats.layer1_cycles + perf_stats.layer2_cycles;
    
    // ------------------------------------------------------------------
    // Post-processing: Softmax and Prediction
    // ------------------------------------------------------------------
    printf("Softmax and prediction\n");
    
    // Compute softmax probabilities
    if (probabilities != NULL) {
        mnist_softmax(network_output, probabilities, OUTPUT_SIZE);
    }
    
    // Get predicted class
    int predicted_class = mnist_argmax(network_output, OUTPUT_SIZE);
    
    printf("\n=== Inference Complete ===\n");
    printf("Total cycles: %u\n", perf_stats.total_cycles);
    
    return predicted_class;
}

// =============================================================================
// Print Results
// =============================================================================

void mnist_print_prediction(int predicted_class, const float* probabilities) {
    printf("\n==========================================\n");
    printf("         MNIST Prediction Results         \n");
    printf("==========================================\n");
    printf("Predicted Digit: %d\n", predicted_class);
    printf("\nClass Probabilities:\n");
    printf("------------------------------------------\n");
    
    for (int i = 0; i < OUTPUT_SIZE; i++) {
        printf("  Digit %d: %6.2f%%", i, probabilities[i] * 100.0f);
        
        // Visual bar chart
        int bar_length = (int)(probabilities[i] * 50);
        printf(" |");
        for (int j = 0; j < bar_length; j++) {
            printf("█");
        }
        printf("\n");
    }
    
    printf("==========================================\n");
}

// =============================================================================
// Performance Statistics
// =============================================================================

void mnist_print_performance(void) {
    printf("\n==========================================\n");
    printf("       Performance Statistics             \n");
    printf("==========================================\n");
    printf("Layer 1 (784→128): %10u cycles\n", perf_stats.layer1_cycles);
    printf("Layer 2 (128→10):  %10u cycles\n", perf_stats.layer2_cycles);
    printf("------------------------------------------\n");
    printf("Total Inference:   %10u cycles\n", perf_stats.total_cycles);
    
    // Assuming 100 MHz clock
    float time_ms = (float)perf_stats.total_cycles / 100000.0f;
    printf("Time @ 100MHz:     %10.2f ms\n", time_ms);
    printf("Throughput:        %10.2f inferences/sec\n", 1000.0f / time_ms);
    printf("==========================================\n");
}

// =============================================================================
// Example Test Image (5×5 downsampled "7" for demo)
// =============================================================================

static const uint8_t test_image_7[INPUT_SIZE] = {
    // Simplified 28×28 image of digit "7"
    // Top rows with horizontal line
    0,   0,   0,   0,   0,   255, 255, 255, 255, 255, 255, 255, 255, 255, 0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
    0,   0,   0,   0,   0,   255, 255, 255, 255, 255, 255, 255, 255, 255, 0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
    0,   0,   0,   0,   0,   255, 255, 255, 255, 255, 255, 255, 255, 255, 0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
    0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   255, 255, 255, 0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
    0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   255, 255, 255, 0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
    0,   0,   0,   0,   0,   0,   0,   0,   0,   255, 255, 255, 0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
    0,   0,   0,   0,   0,   0,   0,   0,   255, 255, 255, 0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
    0,   0,   0,   0,   0,   0,   0,   255, 255, 255, 0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
    0,   0,   0,   0,   0,   0,   255, 255, 255, 0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
    0,   0,   0,   0,   0,   255, 255, 255, 0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
    // Continue diagonal line
    0,   0,   0,   0,   255, 255, 255, 0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
    0,   0,   0,   255, 255, 255, 0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
    0,   0,   255, 255, 255, 0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
    0,   0,   255, 255, 255, 0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
    0,   0,   255, 255, 0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
    0,   0,   255, 255, 0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
    0,   0,   255, 255, 0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
    0,   0,   255, 255, 0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
    0,   0,   255, 255, 0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
    0,   0,   255, 255, 0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
    0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
    0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
    0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
    0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
    0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
    0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
    0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
    0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
};

// =============================================================================
// Main Function - Demo Application
// =============================================================================

int main(void) {
    printf("\n");
    printf("========================================\n");
    printf("  NeuroRISC MNIST Inference Demo\n");
    printf("  2-Layer Fully Connected Network\n");
    printf("  784 → 128 → 10\n");
    printf("========================================\n\n");
    
    // Initialize NPU hardware
    npu_init();
    printf("NPU initialized\n\n");
    
    // Load model weights
    mnist_init_model();
    printf("\n");
    
    // Run inference on test image
    float probabilities[OUTPUT_SIZE];
    int prediction = mnist_infer(test_image_7, probabilities);
    
    // Print results
    mnist_print_prediction(prediction, probabilities);
    mnist_print_performance();
    
    printf("\n✓ MNIST inference complete!\n\n");
    
    return 0;
}
