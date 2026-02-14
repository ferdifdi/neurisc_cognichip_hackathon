# =============================================================================
# Makefile for NeuroRISC Software
# =============================================================================

# Toolchain configuration (RISC-V GCC)
CC = riscv32-unknown-elf-gcc
AR = riscv32-unknown-elf-ar
OBJCOPY = riscv32-unknown-elf-objcopy
OBJDUMP = riscv32-unknown-elf-objdump
SIZE = riscv32-unknown-elf-size

# Compiler flags
CFLAGS = -march=rv32im -mabi=ilp32
CFLAGS += -O2 -g
CFLAGS += -Wall -Wextra
CFLAGS += -ffunction-sections -fdata-sections
CFLAGS += -I.

# Linker flags
LDFLAGS = -march=rv32im -mabi=ilp32
LDFLAGS += -Wl,--gc-sections
LDFLAGS += -nostartfiles -nostdlib
LDFLAGS += -T linker.ld

# Source files
RUNTIME_SRCS = neurisc_runtime.c
MNIST_SRCS = mnist_inference.c

# Object files
RUNTIME_OBJS = $(RUNTIME_SRCS:.c=.o)
MNIST_OBJS = $(MNIST_SRCS:.c=.o)

# Output files
RUNTIME_LIB = libneurisc.a
MNIST_ELF = mnist_inference.elf
MNIST_BIN = mnist_inference.bin
MNIST_HEX = mnist_inference.hex

# Default target
all: $(RUNTIME_LIB) $(MNIST_ELF) $(MNIST_BIN) $(MNIST_HEX)
	@echo "Build complete!"
	@$(SIZE) $(MNIST_ELF)

# Runtime library
$(RUNTIME_LIB): $(RUNTIME_OBJS)
	@echo "Creating runtime library..."
	$(AR) rcs $@ $^

# MNIST inference executable
$(MNIST_ELF): $(MNIST_OBJS) $(RUNTIME_LIB)
	@echo "Linking MNIST inference application..."
	$(CC) $(LDFLAGS) -o $@ $(MNIST_OBJS) -L. -lneurisc

# Binary file (for loading to FPGA/ASIC)
$(MNIST_BIN): $(MNIST_ELF)
	@echo "Generating binary file..."
	$(OBJCOPY) -O binary $< $@

# Hex file (for memory initialization)
$(MNIST_HEX): $(MNIST_ELF)
	@echo "Generating hex file..."
	$(OBJCOPY) -O ihex $< $@

# Compile C files
%.o: %.c
	@echo "Compiling $<..."
	$(CC) $(CFLAGS) -c -o $@ $<

# Disassembly for debugging
disasm: $(MNIST_ELF)
	$(OBJDUMP) -d $< > mnist_inference.asm

# Clean build artifacts
clean:
	rm -f *.o *.a *.elf *.bin *.hex *.asm
	@echo "Clean complete!"

# Help target
help:
	@echo "NeuroRISC Software Build System"
	@echo "================================"
	@echo ""
	@echo "Targets:"
	@echo "  all      - Build everything (default)"
	@echo "  clean    - Remove build artifacts"
	@echo "  disasm   - Generate disassembly listing"
	@echo "  help     - Show this help message"
	@echo ""
	@echo "Outputs:"
	@echo "  $(RUNTIME_LIB) - Runtime library"
	@echo "  $(MNIST_ELF)   - MNIST inference executable (ELF)"
	@echo "  $(MNIST_BIN)   - Binary image"
	@echo "  $(MNIST_HEX)   - Intel HEX format"

.PHONY: all clean disasm help
