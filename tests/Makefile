# Build all test binaries for rocm-polaris.
# Run `make` after any hsa-rocr or hip-runtime-amd library rebuild.

HIPCC     := hipcc
HIPFLAGS  := --offload-arch=gfx803
CC        := gcc
CXX       := g++
CFLAGS    := -O2 -I/opt/rocm/include
CXXFLAGS  := -O2 -I/opt/rocm/include
LDFLAGS   := -L/opt/rocm/lib -Wl,-rpath,/opt/rocm/lib
HSA_LIBS  := -lhsa-runtime64

# HIP sources (compiled with hipcc)
HIP_SRC := $(wildcard hip_*.cpp) $(wildcard test_gpu_read.cpp test_h2d_boundary.cpp \
           test_h2d_kernel.cpp test_memset_verify.cpp)
HIP_BIN := $(HIP_SRC:.cpp=)

# HSA C++ sources (compiled with g++ + libhsa)
HSA_CXX_SRC := test_interrupt_signal.cpp
HSA_CXX_BIN := $(HSA_CXX_SRC:.cpp=)

# HSA C sources (compiled with gcc + libhsa)
HSA_C_SRC := $(wildcard hsa_*.c)
HSA_C_BIN := $(HSA_C_SRC:.c=)

ALL_BIN := $(HIP_BIN) $(HSA_CXX_BIN) $(HSA_C_BIN)

.PHONY: all clean

all: $(ALL_BIN)
	@echo "Built $(words $(ALL_BIN)) test binaries."

# HIP binaries
$(HIP_BIN): %: %.cpp
	$(HIPCC) $(HIPFLAGS) -o $@ $<

# HSA C++ binaries
$(HSA_CXX_BIN): %: %.cpp
	$(CXX) $(CXXFLAGS) -o $@ $< $(LDFLAGS) $(HSA_LIBS)

# HSA C binaries
$(HSA_C_BIN): %: %.c
	$(CC) $(CFLAGS) -o $@ $< $(LDFLAGS) $(HSA_LIBS)

clean:
	rm -f $(ALL_BIN)
