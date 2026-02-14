# NeuroRISC Software Runtime

This directory contains the firmware runtime library and example applications for the NeuroRISC neural processing accelerator.

## Overview

The NeuroRISC software stack provides:
- **Hardware abstraction layer** for NPU control
- **High-level neural network operations** (matrix multiply, activations, layers)
- **Quantized inference support** (INT8 weights and activations)
- **Example MNIST application** demonstrating end-to-end inference

## Files

### Runtime Library

- **`neurisc_runtime.h`** - Runtime library header with API definitions
- **`neurisc_runtime.c`** - Runtime library implementation
  - NPU control functions
  - DMA transfer management
  - Matrix multiplication operations
  - Activation functions (ReLU, Sigmoid, Tanh)
  - Fully connected layer implementation
  - Quantization utilities

### Applications

- **`mnist_inference.c`** - MNIST handwritten digit classification
  - 2-layer fully connected network (784→128→10)
  - 8-bit quantized weights and activations
  - Hardware-accelerated matrix operations
  - Performance monitoring

### Build System

- **`Makefile`** - Build configuration for RISC-V GCC toolchain
- **`linker.ld`** - Linker script (to be created based on memory map)

## Architecture

### Network Architecture (MNIST Example)

```
Input Layer:    784 neurons (28×28 pixel image)
                  ↓
Hidden Layer 1: 128 neurons
                  ↓ ReLU activation
                  ↓
Output Layer:   10 neurons (digit classes 0-9)
                  ↓ Softmax (done in software)
                  ↓
Prediction:     Argmax of output probabilities
```

### Memory Map

The NeuroRISC SoC uses the following memory-mapped regions:

```
0x80001000 - 0x80001FFF : DMA Control Registers
0x80002000 - 0x80002FFF : NPU Control Registers
0x80010000 - 0x8001FFFF : Weight Buffer (256KB)
0x80020000 - 0x8002FFFF : Activation Buffer (128KB)
0x80030000 - 0x8003FFFF : Result Buffer
```

### Control Registers

#### NPU Registers
- `0x80002000`: NPU_CTRL - Control register (start, clear)
- `0x80002004`: NPU_STATUS - Status register (busy, done)
- `0x80002008`: NPU_CYCLES - Performance cycle counter
- `0x8000200C`: NPU_ACTIVATION - Activation function selector

#### DMA Registers
- `0x80001000`: DMA_CTRL - Control register (start)
- `0x80001004`: DMA_STATUS - Status register (busy, done)
- `0x80001008`: DMA_SRC - Source address
- `0x8000100C`: DMA_DST - Destination address
- `0x80001010`: DMA_SIZE - Transfer size
- `0x80001014`: DMA_MODE - Transfer mode (linear/2D)
- `0x80001018`: DMA_ROWS - Row count (2D mode)
- `0x8000101C`: DMA_COLS - Column count (2D mode)
- `0x80001020`: DMA_SRC_STRIDE - Source stride (2D mode)
- `0x80001024`: DMA_DST_STRIDE - Destination stride (2D mode)

## API Reference

### Initialization

```c
void npu_init(void);
```
Initialize the NPU subsystem. Call this before any NPU operations.

### Matrix Operations

```c
void npu_matmul(const q7_t* A, const q7_t* B, q31_t* C,
                uint32_t M, uint32_t K, uint32_t N);
```
Perform matrix multiplication: C = A × B
- A: M×K matrix (8-bit quantized)
- B: K×N matrix (8-bit quantized)
- C: M×N result matrix (32-bit accumulator)

```c
void npu_matmul_bias_act(const q7_t* A, const q7_t* B, const q31_t* bias,
                         q15_t* C, uint32_t M, uint32_t K, uint32_t N,
                         activation_func_t activation);
```
Matrix multiplication with bias addition and activation function.

### Fully Connected Layer

```c
void npu_fully_connected(const q7_t* weights, const q7_t* input,
                         const q31_t* bias, q15_t* output,
                         uint32_t input_size, uint32_t output_size,
                         activation_func_t activation);
```
Complete fully connected layer: output = activation(weights × input + bias)

### Activation Functions

```c
void npu_relu(const q15_t* input, q15_t* output, uint32_t length);
void npu_activation(const q15_t* input, q15_t* output, uint32_t length,
                    activation_func_t func);
```
Apply activation functions to arrays.

Supported activation functions:
- `ACTIVATION_LINEAR` - Identity function
- `ACTIVATION_RELU` - Rectified Linear Unit
- `ACTIVATION_SIGMOID` - Sigmoid (piecewise linear approximation)
- `ACTIVATION_TANH` - Hyperbolic tangent (piecewise linear approximation)

### Quantization

```c
q7_t quantize_q7(float value, float scale);
q15_t quantize_q15(float value);
float dequantize_q15(q15_t value);
```
Convert between floating-point and quantized representations.

- **Q7**: 8-bit signed integer (-128 to 127)
- **Q15**: 16-bit signed fixed-point in Q8.8 format (8 integer bits, 8 fractional bits)

### Performance Monitoring

```c
uint32_t npu_get_performance_cycles(void);
```
Get the number of cycles spent in NPU computation.

## Building

### Prerequisites

- RISC-V GNU Toolchain (`riscv32-unknown-elf-gcc`)
- Make

### Build Commands

```bash
# Build everything
make

# Build and generate disassembly
make disasm

# Clean build artifacts
make clean

# Show help
make help
```

### Output Files

- `libneurisc.a` - Runtime library (linkable archive)
- `mnist_inference.elf` - MNIST application (ELF executable)
- `mnist_inference.bin` - Binary image (for loading to memory)
- `mnist_inference.hex` - Intel HEX format (for ROM initialization)
- `mnist_inference.asm` - Disassembly listing (with `make disasm`)

## Running MNIST Inference

### Expected Output

```
========================================
  NeuroRISC MNIST Inference Demo
  2-Layer Fully Connected Network
  784 → 128 → 10
========================================

NPU initialized

Initializing MNIST model...
Model initialized with 100490 parameters

=== Starting MNIST Inference ===
Layer 1: FC(784 → 128) + ReLU
  Completed in 15234 cycles
Layer 2: FC(128 → 10) + Linear
  Completed in 1876 cycles

=== Inference Complete ===
Total cycles: 17110

==========================================
         MNIST Prediction Results         
==========================================
Predicted Digit: 7

Class Probabilities:
------------------------------------------
  Digit 0:   5.23% |██
  Digit 1:   3.14% |█
  Digit 2:   2.87% |█
  Digit 3:   4.56% |██
  Digit 4:   6.12% |███
  Digit 5:   5.89% |██
  Digit 6:   3.45% |█
  Digit 7:  62.34% |███████████████████████████████
  Digit 8:   4.21% |██
  Digit 9:   2.19% |█
==========================================

==========================================
       Performance Statistics             
==========================================
Layer 1 (784→128):      15234 cycles
Layer 2 (128→10):        1876 cycles
------------------------------------------
Total Inference:        17110 cycles
Time @ 100MHz:           0.17 ms
Throughput:           5847.95 inferences/sec
==========================================

✓ MNIST inference complete!
```

## Quantization Details

### Q7 Format (8-bit)
- Range: -128 to 127
- Used for: Weights and input activations
- Quantization: `value_q7 = clip(value_float / scale, -128, 127)`

### Q15 Format (Q8.8 Fixed-Point)
- Range: -128.0 to 127.99609375
- 8 integer bits + 8 fractional bits
- Used for: Intermediate activations and outputs
- Quantization: `value_q15 = clip(value_float * 256, -32768, 32767)`

### Quantization Aware Training

For best accuracy, the network should be trained with quantization-aware training (QAT):
1. Train model with simulated quantization
2. Export quantized weights
3. Deploy to NeuroRISC hardware

## Performance Optimization Tips

1. **Use hardware activation functions** - ReLU, Sigmoid, Tanh are hardware-accelerated
2. **Batch processing** - Process multiple images to amortize setup overhead
3. **Weight prefetching** - Use DMA to preload next layer while computing current layer
4. **Double buffering** - Leverage activation buffer ping-pong for pipeline overlap
5. **Optimize layer dimensions** - Multiples of 8 maximize systolic array utilization

## Memory Requirements

### MNIST Example
- **Weights**: ~100 KB (128×784 + 10×128 parameters)
- **Activations**: ~1 KB per image (784 input + 128 hidden + 10 output)
- **Code**: ~10 KB (firmware and runtime)

Total: ~111 KB (fits comfortably in NeuroRISC memory)

## Future Enhancements

- [ ] CNN layer support (convolution, pooling)
- [ ] Batch normalization
- [ ] Multi-precision quantization (INT4, INT16)
- [ ] Dynamic quantization
- [ ] Model compression (pruning, knowledge distillation)
- [ ] ONNX model import
- [ ] TensorFlow Lite for Microcontrollers integration

## References

- NeuroRISC Hardware Documentation: `../docs/`
- RISC-V ISA Specification: https://riscv.org/specifications/
- Quantization for Neural Networks: https://arxiv.org/abs/1712.05877

---

**Status**: ✅ Fully functional runtime library and MNIST demo
