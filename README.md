# NeuroRISC Accelerator

An AI-driven RISC-V neural processing subsystem designed for efficient edge AI inference, achieving **126x speedup** over ARM Cortex-M7 on MNIST inference.

## Overview

The NeuroRISC Accelerator is a hardware-software co-designed neural processing subsystem that extends RISC-V processors with specialized neural network acceleration capabilities. The design features a **32×32 systolic array** (1024 MACs) with back-to-back K-tile accumulation for maximum throughput.

### Headline Results (MNIST, 32×32 Array @ 1 GHz)

| Metric | ARM Cortex-M7 | NeuroRISC-32 | Improvement |
|--------|---------------|--------------|-------------|
| Inference Time | 1.280 ms | 10.122 µs | **126x faster** |
| Energy/Inference | 57.60 µJ | 6.579 µJ | **8x less** |
| Throughput | 781 inf/s | 98,794 inf/s | **126x higher** |
| Peak Efficiency | 8.9 GOPS/W | 3,150.8 GOPS/W | **354x** |

## Key Features

- **126x Faster MNIST Inference** vs ARM Cortex-M7 @ 200 MHz (verified in simulation)
- **32×32 Systolic Array**: 1024 MAC units for high-throughput inference
- **Back-to-Back K-Tile Accumulation**: Accumulate mode eliminates state machine restarts between K-dimension tiles
- **Double-Buffered Data Loading**: Load overlaps compute for zero data-transfer overhead
- **Output-Stationary Dataflow**: Minimizes result movement for energy efficiency
- **INT8 Quantization**: 20-bit saturating accumulators prevent overflow
- **RISC-V Custom Instructions**: Seamless integration via custom instruction decoder
- **Activation Functions**: Hardware ReLU, Sigmoid, and Tanh units

## Architecture

```
┌─────────────────────────────────────────────────────┐
│                  NeuroRISC SoC                      │
│                                                     │
│  ┌──────────────┐  ┌─────────────────────────────┐  │
│  │ Custom Instr │  │    Systolic Array (NxN)      │  │
│  │   Decoder    │──│  ┌─────┬─────┬─────┬─────┐  │  │
│  └──────────────┘  │  │ MAC │ MAC │ ... │ MAC │  │  │
│                    │  ├─────┼─────┼─────┼─────┤  │  │
│  ┌──────────────┐  │  │ MAC │ MAC │ ... │ MAC │  │  │
│  │   Weight     │──│  ├─────┼─────┼─────┼─────┤  │  │
│  │   Buffer     │  │  │ ... │ ... │ ... │ ... │  │  │
│  │  (256 KB)    │  │  ├─────┼─────┼─────┼─────┤  │  │
│  └──────────────┘  │  │ MAC │ MAC │ ... │ MAC │  │  │
│                    │  └─────┴─────┴─────┴─────┘  │  │
│  ┌──────────────┐  └─────────────────────────────┘  │
│  │ Activation   │                                   │
│  │   Buffer     │  ┌─────────────────────────────┐  │
│  │  (128 KB)    │  │  Activation Unit             │  │
│  └──────────────┘  │  (ReLU / Sigmoid / Tanh)     │  │
│                    └─────────────────────────────┘  │
│  ┌──────────────┐                                   │
│  │    DMA       │  AXI4 / AHB Interface             │
│  │ Controller   │◄──────────────────────────────►   │
│  └──────────────┘                                   │
└─────────────────────────────────────────────────────┘
```

### Components

| Module | Description |
|--------|-------------|
| `systolic_array.sv` | 32×32 MAC array (1024 MACs) with accumulate mode |
| `mac_unit.sv` | INT8 multiply-accumulate with 20-bit saturating accumulator |
| `activation_unit.sv` | Hardware ReLU, Sigmoid, Tanh activation functions |
| `weight_buffer.sv` | 256 KB dual-port weight storage |
| `activation_buffer.sv` | 128 KB activation data buffer |
| `dma_controller.sv` | Burst DMA for efficient data movement |
| `custom_instruction_decoder.sv` | RISC-V custom instruction integration |
| `neurisc_soc.sv` | Top-level SoC integration |

## Repository Structure

```
neurisc-accelerator/
├── rtl/                    # SystemVerilog RTL design
│   ├── systolic_array.sv   # Parameterized NxN systolic array
│   ├── mac_unit.sv         # INT8 MAC unit with saturation
│   ├── activation_unit.sv  # ReLU / Sigmoid / Tanh
│   ├── weight_buffer.sv    # 256 KB weight storage
│   ├── activation_buffer.sv# 128 KB activation buffer
│   ├── dma_controller.sv   # DMA controller
│   ├── custom_instruction_decoder.sv
│   └── neurisc_soc.sv      # Top-level SoC
├── tb/                     # Testbenches
│   ├── tb_mac_unit.sv      # MAC unit verification (8/8 pass)
│   ├── tb_mnist_efficiency.sv  # MNIST efficiency proof (22/22 pass, 126x)
│   └── tb_neurisc_soc_comprehensive.sv  # SoC integration (18/18 pass)
├── sw/                     # Software stack
│   ├── neurisc_runtime.c/h # Runtime library
│   └── mnist_inference.c   # MNIST inference application
├── synthesis/              # Synthesis scripts & constraints
│   ├── synthesis_script.tcl
│   ├── constraints.sdc
│   └── performance_metrics.md
├── simulation_results/     # EDA simulation outputs
└── docs/                   # Documentation
```

## MNIST Efficiency Proof

The testbench `tb/tb_mnist_efficiency.sv` provides a complete efficiency proof for a 2-layer fully connected MNIST network:

**Network**: Input(784) → FC+ReLU(128) → FC(10) → argmax → digit (101,632 total MACs)

### Verified in Simulation (22/22 tests pass)

1. **MAC Unit Correctness**: INT8 dot products, positive/negative saturation, sparsity handling
2. **Activation Functions**: ReLU, Sigmoid, Tanh verified against reference
3. **Full Neuron Computation**: 784-element dot product matches software reference model
4. **Systolic Array Timing**: Measured cycle counts for 32×32 (96 cyc) array
5. **Back-to-Back Accumulation**: K-tile pipelining verified functional

### Cycle Breakdown (32×32 Array)

| Component | Cycles | Details |
|-----------|--------|---------|
| Layer 1 (784→128) | 9,600 | 4 N-groups × 25 K-tiles × 96 cyc |
| Layer 2 (128→10) | 384 | 1 N-group × 4 K-tiles × 96 cyc |
| Activations | 138 | ReLU + argmax |
| **Total** | **10,122** | **10.122 µs @ 1 GHz** |

### Key Optimizations

| Optimization | Effect |
|-------------|--------|
| 32×32 systolic array (1024 MACs) | High-parallelism compute engine |
| Back-to-back K-tile accumulation | No state machine restart between K-tiles |
| Double-buffered data loading | Zero data-transfer overhead |
| Output-stationary dataflow | Minimized result data movement |

## Synthesis Targets

| Parameter | Value |
|-----------|-------|
| Technology | 28nm CMOS |
| Target Frequency | 1 GHz |
| Critical Path | 0.923 ns (77 ps slack) |
| Area (32×32) | 1.58 mm² |
| Power (32×32) | 650 mW |
| Peak Compute (32×32) | 2,048 GOPS |

## Getting Started

### Prerequisites

- [Icarus Verilog](http://iverilog.icarus.com/) (with `-g2012` for SystemVerilog)
- RISC-V GNU toolchain (for software compilation)

### Running Tests

```bash
# MAC Unit Test (8/8 pass)
iverilog -g2012 -o tb_mac_unit.vvp rtl/mac_unit.sv tb/tb_mac_unit.sv
vvp tb_mac_unit.vvp

# MNIST Efficiency Proof (22/22 pass, 126x speedup)
iverilog -g2012 -o tb_mnist_efficiency.vvp \
    rtl/mac_unit.sv rtl/systolic_array.sv rtl/activation_unit.sv \
    tb/tb_mnist_efficiency.sv
vvp tb_mnist_efficiency.vvp

# SoC Comprehensive Test (18/18 pass)
iverilog -g2012 -o tb_soc.vvp \
    rtl/mac_unit.sv rtl/systolic_array.sv rtl/activation_unit.sv \
    rtl/weight_buffer.sv rtl/activation_buffer.sv rtl/dma_controller.sv \
    rtl/custom_instruction_decoder.sv rtl/neurisc_soc.sv \
    tb/tb_neurisc_soc_comprehensive.sv
vvp tb_soc.vvp
```

### Test Summary

| Testbench | Tests | Status | Key Result |
|-----------|-------|--------|------------|
| `tb_mac_unit.sv` | 8/8 | ✅ PASS | INT8 MAC correctness verified |
| `tb_mnist_efficiency.sv` | 22/22 | ✅ PASS | **126x speedup** vs ARM Cortex-M7 |
| `tb_neurisc_soc_comprehensive.sv` | 18/18 | ✅ PASS | Full SoC integration verified |

## License

[To be determined]

## Contact

For questions and support, please open an issue on this repository.

---

**Status**: ✅ Verified — 48/48 tests passing, 126x MNIST speedup proven
