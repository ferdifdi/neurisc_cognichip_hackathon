// =============================================================================
// File: neurisc_runtime.h
// Description: NeuroRISC Runtime Library Header
// 
// Provides high-level API for NPU operations including matrix multiplication,
// activation functions, and convolution layers.
// =============================================================================

#ifndef NEURISC_RUNTIME_H
#define NEURISC_RUNTIME_H

#include <stdint.h>
#include <stddef.h>

// =============================================================================
// Memory-Mapped Register Addresses (from neurisc_soc.sv)
// =============================================================================

// NPU Control Registers
#define NPU_CTRL_REG       ((volatile uint32_t*)0x80002000)
#define NPU_STATUS_REG     ((volatile uint32_t*)0x80002004)
#define NPU_CYCLES_REG     ((volatile uint32_t*)0x80002008)
#define NPU_ACTIVATION_REG ((volatile uint32_t*)0x8000200C)

// DMA Control Registers
#define DMA_CTRL_REG       ((volatile uint32_t*)0x80001000)
#define DMA_STATUS_REG     ((volatile uint32_t*)0x80001004)
#define DMA_SRC_REG        ((volatile uint32_t*)0x80001008)
#define DMA_DST_REG        ((volatile uint32_t*)0x8000100C)
#define DMA_SIZE_REG       ((volatile uint32_t*)0x80001010)
#define DMA_MODE_REG       ((volatile uint32_t*)0x80001014)
#define DMA_ROWS_REG       ((volatile uint32_t*)0x80001018)
#define DMA_COLS_REG       ((volatile uint32_t*)0x8000101C)
#define DMA_SRC_STRIDE_REG ((volatile uint32_t*)0x80001020)
#define DMA_DST_STRIDE_REG ((volatile uint32_t*)0x80001024)

// Memory Buffers
#define WEIGHT_BUFFER_BASE  0x80010000
#define ACTIVATION_BUFFER_BASE 0x80020000
#define RESULT_BUFFER_BASE  0x80030000

// =============================================================================
// Control Register Bit Definitions
// =============================================================================

// NPU Control bits
#define NPU_CTRL_START  (1 << 0)
#define NPU_CTRL_CLEAR  (1 << 1)

// NPU Status bits
#define NPU_STATUS_DONE (1 << 0)
#define NPU_STATUS_BUSY (1 << 1)

// DMA Control bits
#define DMA_CTRL_START  (1 << 0)

// DMA Status bits
#define DMA_STATUS_DONE (1 << 0)
#define DMA_STATUS_BUSY (1 << 1)

// DMA Mode bits
#define DMA_MODE_LINEAR (0 << 0)
#define DMA_MODE_2D     (1 << 0)

// =============================================================================
// Activation Function Types
// =============================================================================

typedef enum {
    ACTIVATION_LINEAR  = 0,
    ACTIVATION_RELU    = 1,
    ACTIVATION_SIGMOID = 2,
    ACTIVATION_TANH    = 3
} activation_func_t;

// =============================================================================
// Data Types
// =============================================================================

typedef int8_t  q7_t;   // 8-bit quantized values
typedef int16_t q15_t;  // 16-bit quantized values (Q8.8 fixed-point)
typedef int32_t q31_t;  // 32-bit accumulator

// Matrix descriptor
typedef struct {
    uint32_t rows;
    uint32_t cols;
    void*    data;
} matrix_t;

// =============================================================================
// Low-Level Hardware Control Functions
// =============================================================================

// NPU control
void npu_start_computation(void);
void npu_clear_accumulators(void);
void npu_wait_done(void);
uint32_t npu_get_cycles(void);
void npu_set_activation(activation_func_t func);

// DMA control
void dma_transfer_linear(uint32_t src, uint32_t dst, uint16_t size);
void dma_transfer_2d(uint32_t src, uint32_t dst, 
                     uint16_t rows, uint16_t cols,
                     uint16_t src_stride, uint16_t dst_stride);
void dma_wait_done(void);

// =============================================================================
// High-Level NPU Operations
// =============================================================================

// Matrix multiplication: C = A × B
// A: M×K, B: K×N, C: M×N (max 8×8 per operation)
void npu_matmul(const q7_t* A, const q7_t* B, q31_t* C,
                uint32_t M, uint32_t K, uint32_t N);

// Matrix multiplication with bias and activation
void npu_matmul_bias_act(const q7_t* A, const q7_t* B, const q31_t* bias,
                         q15_t* C, uint32_t M, uint32_t K, uint32_t N,
                         activation_func_t activation);

// Apply activation function to array
void npu_activation(const q15_t* input, q15_t* output, uint32_t length,
                    activation_func_t func);

// ReLU activation
void npu_relu(const q15_t* input, q15_t* output, uint32_t length);

// Quantize float to Q7 (8-bit)
q7_t quantize_q7(float value, float scale);

// Quantize float to Q15 (Q8.8 fixed-point)
q15_t quantize_q15(float value);

// Dequantize Q15 to float
float dequantize_q15(q15_t value);

// Fully connected layer: y = Wx + b with activation
void npu_fully_connected(const q7_t* weights, const q7_t* input,
                         const q31_t* bias, q15_t* output,
                         uint32_t input_size, uint32_t output_size,
                         activation_func_t activation);

// 2D Convolution layer (simplified for 8×8 systolic array)
void npu_conv2d(const q7_t* input, const q7_t* kernel, q31_t* output,
                uint32_t input_h, uint32_t input_w, uint32_t input_ch,
                uint32_t kernel_size, uint32_t output_ch,
                uint32_t stride, uint32_t padding);

// =============================================================================
// Utility Functions
// =============================================================================

// Memory copy using DMA
void npu_memcpy(void* dst, const void* src, size_t size);

// Initialize NPU subsystem
void npu_init(void);

// Get NPU performance statistics
uint32_t npu_get_performance_cycles(void);

#endif // NEURISC_RUNTIME_H
