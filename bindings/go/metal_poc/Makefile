# Makefile for Metal GPU acceleration proof of concept
# For c-kzg-4844 on Apple M1

CC = clang
CXX = clang++
METALC = xcrun -sdk macosx metal
METALLIB = xcrun -sdk macosx metallib

# Paths
KZG_ROOT = ../../..
SRC_DIR = $(KZG_ROOT)/src
LIB_DIR = $(KZG_ROOT)/lib
BLST_DIR = $(KZG_ROOT)/blst

# Compiler flags
CFLAGS = -O3 -Wall -I$(SRC_DIR) -I$(BLST_DIR)/bindings
CXXFLAGS = $(CFLAGS) -std=c++17
OBJCFLAGS = -fobjc-arc -fmodules
LDFLAGS = -L$(LIB_DIR) -L$(BLST_DIR) -framework Metal -framework Foundation -framework CoreGraphics

# Libraries
LIBS = -lblst -lm

# Source files
METAL_SOURCES = bls12_381_field.metal
OBJC_SOURCES = MetalKZGAccelerator.mm
C_SOURCES = kzg_metal_integration.c benchmark_metal.c

# Object files
METAL_LIB = bls12_381_field.metallib
OBJECTS = MetalKZGAccelerator.o kzg_metal_integration.o benchmark_metal.o

# KZG library files (simplified list)
KZG_SOURCES = $(SRC_DIR)/ckzg.c \
              $(SRC_DIR)/common/alloc.c \
              $(SRC_DIR)/common/bytes.c \
              $(SRC_DIR)/common/ec.c \
              $(SRC_DIR)/common/fr.c \
              $(SRC_DIR)/common/lincomb.c \
              $(SRC_DIR)/common/utils.c \
              $(SRC_DIR)/eip4844/blob.c \
              $(SRC_DIR)/eip4844/eip4844.c \
              $(SRC_DIR)/eip7594/cell.c \
              $(SRC_DIR)/eip7594/eip7594.c \
              $(SRC_DIR)/eip7594/fft.c \
              $(SRC_DIR)/eip7594/fk20.c \
              $(SRC_DIR)/eip7594/poly.c \
              $(SRC_DIR)/eip7594/recovery.c \
              $(SRC_DIR)/setup/setup.c

KZG_OBJECTS = $(KZG_SOURCES:.c=.o)

# Targets
.PHONY: all clean run test

all: benchmark_metal

# Compile Metal shaders
$(METAL_LIB): $(METAL_SOURCES)
	@echo "Compiling Metal shaders..."
	$(METALC) -c $< -o bls12_381_field.air
	$(METALLIB) bls12_381_field.air -o $@
	@rm -f bls12_381_field.air

# Compile Objective-C++ wrapper
MetalKZGAccelerator.o: MetalKZGAccelerator.mm MetalKZGAccelerator.h
	@echo "Compiling Objective-C++ wrapper..."
	$(CXX) $(CXXFLAGS) $(OBJCFLAGS) -c $< -o $@

# Compile integration layer
kzg_metal_integration.o: kzg_metal_integration.c MetalKZGAccelerator.h
	@echo "Compiling integration layer..."
	$(CC) $(CFLAGS) -c $< -o $@

# Compile benchmark
benchmark_metal.o: benchmark_metal.c
	@echo "Compiling benchmark..."
	$(CC) $(CFLAGS) -c $< -o $@

# Compile KZG sources
%.o: %.c
	@echo "Compiling $<..."
	$(CC) $(CFLAGS) -c $< -o $@

# Link everything
benchmark_metal: $(OBJECTS) $(KZG_OBJECTS) $(METAL_LIB)
	@echo "Linking benchmark_metal..."
	$(CXX) $(OBJECTS) $(KZG_OBJECTS) $(LDFLAGS) $(LIBS) -o $@
	@echo "Build complete!"

# Run the benchmark
run: benchmark_metal
	@echo "Running benchmark..."
	./benchmark_metal

# Test Metal functionality
test: benchmark_metal
	@echo "Testing Metal GPU acceleration..."
	./benchmark_metal | grep -E "(Speedup|GPU|CPU Time)"

# Clean build artifacts
clean:
	@echo "Cleaning..."
	rm -f $(OBJECTS) $(KZG_OBJECTS) $(METAL_LIB) benchmark_metal
	rm -f *.air *.o
	@echo "Clean complete!"

# Build info
info:
	@echo "=== Build Configuration ==="
	@echo "KZG Root: $(KZG_ROOT)"
	@echo "Compiler: $(CC)"
	@echo "Metal SDK: $(shell xcrun --show-sdk-path)"
	@echo "Architecture: $(shell uname -m)"
	@echo "=========================="

# Help
help:
	@echo "Available targets:"
	@echo "  make all      - Build the benchmark program"
	@echo "  make run      - Build and run the benchmark"
	@echo "  make test     - Run a quick test"
	@echo "  make clean    - Remove build artifacts"
	@echo "  make info     - Show build configuration"
	@echo "  make help     - Show this help message"