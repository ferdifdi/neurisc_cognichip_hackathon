// =============================================================================
// File: neurisc_runtime.c
// Description: NeuroRISC Runtime Library Implementation
// 
// Implements NPU control functions and high-level operations for neural
// network inference on the NeuroRISC accelerator.
// =============================================================================

#include "neurisc_runtime.h"
#include <string.h>

// =============================================================================
// Low-Level Hardware Control Functions
// =============================================================================

void npu_start_computation(void) {
    *NPU_CTRL_REG = NPU_CTRL_START;
}

void npu_clear_accumulators(void) {
    *NPU_CTRL_REG = NPU_CTRL_CLEAR;
}

void npu_wait_done(void) {
    while ((*NPU_STATUS_REG & NPU_STATUS_BUSY) != 0) {
        // Busy-wait for NPU completion
        asm volatile("nop");
    }
}

uint32_t npu_get_cycles(void) {
    return *NPU_CYCLES_REG;
}

void npu_set_activation(activation_func_t func) {
    *NPU_ACTIVATION_REG = (uint32_t)func;
}

void dma_transfer_linear(uint32_t src, uint32_t dst, uint16_t size) {
    *DMA_SRC_REG = src;
    *DMA_DST_REG = dst;
    *DMA_SIZE_REG = size;
    *DMA_MODE_REG = DMA_MODE_LINEAR;
    *DMA_CTRL_REG = DMA_CTRL_START;
}

void dma_transfer_2d(uint32_t src, uint32_t dst,
                     uint16_t rows, uint16_t cols,
                     uint16_t src_stride, uint16_t dst_stride) {
    *DMA_SRC_REG = src;
    *DMA_DST_REG = dst;
    *DMA_ROWS_REG = rows;
    *DMA_COLS_REG = cols;
    *DMA_SRC_STRIDE_REG = src_stride;
    *DMA_DST_STRIDE_REG = dst_stride;
    *DMA_MODE_REG = DMA_MODE_2D;
    *DMA_CTRL_REG = DMA_CTRL_START;
}

void dma_wait_done(void) {
    while ((*DMA_STATUS_REG & DMA_STATUS_BUSY) != 0) {
        // Busy-wait for DMA completion
        asm volatile("nop");
    }
}

// =============================================================================
// Matrix Multiplication: C = A × B
// A: M×K (rows × cols)
// B: K×N (rows × cols)
// C: M×N (rows × cols)
// =============================================================================

void npu_matmul(const q7_t* A, const q7_t* B, q31_t* C,
                uint32_t M, uint32_t K, uint32_t N) {
    
    // Process in 8×8 tiles for the systolic array
    for (uint32_t m = 0; m < M; m += 8) {
        for (uint32_t n = 0; n < N; n += 8) {
            // Clear accumulators
            npu_clear_accumulators();
            
            // Compute tile: process K dimension in blocks of 8
            for (uint32_t k = 0; k < K; k += 8) {
                uint32_t tile_m = (M - m) < 8 ? (M - m) : 8;
                uint32_t tile_k = (K - k) < 8 ? (K - k) : 8;
                uint32_t tile_n = (N - n) < 8 ? (N - n) : 8;
                
                // Load weights (A tile) to weight buffer via DMA
                for (uint32_t i = 0; i < tile_m; i++) {
                    dma_transfer_linear(
                        (uint32_t)&A[(m + i) * K + k],
                        WEIGHT_BUFFER_BASE + i * 8,
                        tile_k
                    );
                    dma_wait_done();
                }
                
                // Load inputs (B tile) to activation buffer via DMA
                for (uint32_t j = 0; j < tile_k; j++) {
                    dma_transfer_linear(
                        (uint32_t)&B[(k + j) * N + n],
                        ACTIVATION_BUFFER_BASE + j * 8,
                        tile_n
                    );
                    dma_wait_done();
                }
                
                // Start NPU computation
                npu_start_computation();
                npu_wait_done();
            }
            
            // Read results from NPU
            for (uint32_t i = 0; i < 8 && (m + i) < M; i++) {
                for (uint32_t j = 0; j < 8 && (n + j) < N; j++) {
                    // In real hardware, read from result registers
                    // For now, we'll simulate reading from result buffer
                    volatile q31_t* result_ptr = (volatile q31_t*)(RESULT_BUFFER_BASE + (i * 8 + j) * 4);
                    C[(m + i) * N + (n + j)] = *result_ptr;
                }
            }
        }
    }
}

// =============================================================================
// Matrix Multiplication with Bias and Activation
// =============================================================================

void npu_matmul_bias_act(const q7_t* A, const q7_t* B, const q31_t* bias,
                         q15_t* C, uint32_t M, uint32_t K, uint32_t N,
                         activation_func_t activation) {
    
    // Temporary buffer for raw results
    static q31_t temp_result[64];  // Max 8×8 tile
    
    // Set activation function
    npu_set_activation(activation);
    
    // Process in 8×8 tiles
    for (uint32_t m = 0; m < M; m += 8) {
        for (uint32_t n = 0; n < N; n += 8) {
            // Perform matrix multiplication for this tile
            npu_matmul(&A[m * K], &B[0], temp_result, 
                      (M - m) < 8 ? (M - m) : 8,
                      K,
                      (N - n) < 8 ? (N - n) : 8);
            
            // Add bias and apply activation
            for (uint32_t i = 0; i < 8 && (m + i) < M; i++) {
                for (uint32_t j = 0; j < 8 && (n + j) < N; j++) {
                    uint32_t idx = (m + i) * N + (n + j);
                    
                    // Add bias if provided
                    q31_t value = temp_result[i * 8 + j];
                    if (bias != NULL) {
                        value += bias[n + j];
                    }
                    
                    // Requantize to Q15 (shift right by 8 bits for Q7×Q7=Q14, then to Q8.8)
                    q15_t quantized = (q15_t)(value >> 6);  // Q14 -> Q8.8
                    
                    // Apply activation (done by hardware activation unit)
                    // For now, software implementation
                    switch (activation) {
                        case ACTIVATION_RELU:
                            C[idx] = (quantized < 0) ? 0 : quantized;
                            break;
                        case ACTIVATION_LINEAR:
                            C[idx] = quantized;
                            break;
                        case ACTIVATION_SIGMOID:
                        case ACTIVATION_TANH:
                            // Use hardware activation unit
                            C[idx] = quantized;  // Would be processed by HW
                            break;
                        default:
                            C[idx] = quantized;
                    }
                }
            }
        }
    }
}

// =============================================================================
// Activation Functions
// =============================================================================

void npu_activation(const q15_t* input, q15_t* output, uint32_t length,
                    activation_func_t func) {
    npu_set_activation(func);
    
    // Process through activation unit
    for (uint32_t i = 0; i < length; i++) {
        // In real hardware, this would use the activation unit
        // For now, software implementation
        switch (func) {
            case ACTIVATION_LINEAR:
                output[i] = input[i];
                break;
                
            case ACTIVATION_RELU:
                output[i] = (input[i] < 0) ? 0 : input[i];
                break;
                
            case ACTIVATION_SIGMOID:
                // Piecewise linear approximation (matching hardware)
                if (input[i] < -2048) output[i] = 0;
                else if (input[i] < -512) output[i] = 64 + ((input[i] + 512) >> 4);
                else if (input[i] < 512) output[i] = 128 + (input[i] >> 2);
                else if (input[i] < 2048) output[i] = 192 + ((input[i] - 512) >> 4);
                else output[i] = 256;
                break;
                
            case ACTIVATION_TANH:
                // Piecewise linear approximation (matching hardware)
                if (input[i] < -1024) output[i] = -256;
                else if (input[i] < -256) output[i] = -256 + ((input[i] + 1024) >> 2);
                else if (input[i] < 256) output[i] = input[i];
                else if (input[i] < 1024) output[i] = 256 - ((1024 - input[i]) >> 2);
                else output[i] = 256;
                break;
        }
    }
}

void npu_relu(const q15_t* input, q15_t* output, uint32_t length) {
    npu_activation(input, output, length, ACTIVATION_RELU);
}

// =============================================================================
// Quantization Functions
// =============================================================================

q7_t quantize_q7(float value, float scale) {
    int32_t quantized = (int32_t)(value / scale);
    if (quantized > 127) quantized = 127;
    if (quantized < -128) quantized = -128;
    return (q7_t)quantized;
}

q15_t quantize_q15(float value) {
    // Q8.8 format: multiply by 256
    int32_t quantized = (int32_t)(value * 256.0f);
    if (quantized > 32767) quantized = 32767;
    if (quantized < -32768) quantized = -32768;
    return (q15_t)quantized;
}

float dequantize_q15(q15_t value) {
    return (float)value / 256.0f;
}

// =============================================================================
// Fully Connected Layer
// =============================================================================

void npu_fully_connected(const q7_t* weights, const q7_t* input,
                         const q31_t* bias, q15_t* output,
                         uint32_t input_size, uint32_t output_size,
                         activation_func_t activation) {
    
    // Reshape input to 1×input_size
    // Weights are output_size×input_size
    // Result is 1×output_size
    
    npu_matmul_bias_act(input, weights, bias, output,
                       1, input_size, output_size, activation);
}

// =============================================================================
// Utility Functions
// =============================================================================

void npu_memcpy(void* dst, const void* src, size_t size) {
    dma_transfer_linear((uint32_t)src, (uint32_t)dst, (uint16_t)size);
    dma_wait_done();
}

void npu_init(void) {
    // Clear NPU state
    npu_clear_accumulators();
    
    // Set default activation to linear
    npu_set_activation(ACTIVATION_LINEAR);
    
    // Wait for any pending operations
    npu_wait_done();
    dma_wait_done();
}

uint32_t npu_get_performance_cycles(void) {
    return npu_get_cycles();
}
