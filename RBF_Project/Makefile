# Makefile for the RBF CUDA Project

# --- Executables ---
EXE = rbf_deformer
TEST_EXE = test_cusolvermp

# --- Compilers ---
# Default to NVHPC 24.9 install (can be overridden by environment variables)
NVHPC24_ROOT ?= /cnic/work/yinxy/Linux_x86_64/24.9
NVHPC_CUDA_VER ?= 12.6
HPCX_VER ?= hpcx-2.20

NVCC = $(NVHPC24_ROOT)/cuda/$(NVHPC_CUDA_VER)/bin/nvcc
CXX = $(NVHPC24_ROOT)/compilers/bin/nvc++

# Use NVHPC HPCX (OpenMPI+UCX+UCC) as a consistent, bundled comm stack.
# This avoids mixing conda/system MPI/UCX/UCC libraries, which often causes crashes
# or cuSOLVERMp returning NOT_SUPPORTED.
HPCX_ROOT ?= $(NVHPC24_ROOT)/comm_libs/$(NVHPC_CUDA_VER)/hpcx/$(HPCX_VER)
MPI_HOME ?= $(HPCX_ROOT)/ompi
MPICC ?= $(MPI_HOME)/bin/mpicc

# --- Source Files ---
RBF_CU_SOURCES = main.cu RBFDeformer.cu
RBF_CPP_SOURCES = MeshLoader.cpp
RBF_OBJECTS = $(RBF_CU_SOURCES:.cu=.o) $(RBF_CPP_SOURCES:.cpp=.o)

TEST_SOURCES = test_cusolvermp.cpp
TEST_OBJECTS = $(TEST_SOURCES:.cpp=.o)

# --- Compiler and Linker Flags ---
# Set GPU architecture targets. Adjusted to include host GPU compute capability (detected automatically).
# For best compatibility we generate SASS for the detected device and include PTX for JIT.
# Example for RTX 40xx (compute 8.9):
ARCH = -gencode=arch=compute_90,code=sm_90 \
	-gencode=arch=compute_89,code=sm_89

# Use C++14 standard and enable optimizations.
MPI_CFLAGS = $(shell $(MPICC) --showme:compile 2>/dev/null || $(MPICC) -showme:compile 2>/dev/null)
MPI_LDFLAGS = -L$(MPI_HOME)/lib -lmpi -pthread
MPI_INC_DIR = $(MPI_HOME)/include

# nvcc does not accept -pthread in flags; remove it for NVCC usage
MPI_CFLAGS_CLEAN = $(filter-out -pthread,$(MPI_CFLAGS))

CUDA_INC_DIR = $(NVHPC24_ROOT)/cuda/$(NVHPC_CUDA_VER)/include
NVHPC_INC_DIR = $(NVHPC24_ROOT)/math_libs/$(NVHPC_CUDA_VER)/include
NCCL_INC_DIR = $(NVHPC24_ROOT)/comm_libs/$(NVHPC_CUDA_VER)/nccl/include

# Include CUDA headers first to avoid conflicting typedefs between system CUDA and NVHPC headers
NVCCFLAGS = -O3 --std=c++14 -DUSE_CUSOLVERMP $(ARCH) -Xcompiler -Wall -I$(CUDA_INC_DIR) -I$(NVHPC_INC_DIR) -I$(NCCL_INC_DIR) -I$(MPI_INC_DIR) $(MPI_CFLAGS_CLEAN)
CXXFLAGS = -O3 --std=c++14 -DUSE_CUSOLVERMP -Wall -I$(CUDA_INC_DIR) -I$(NVHPC_INC_DIR) -I$(NCCL_INC_DIR) -I$(MPI_INC_DIR) $(MPI_CFLAGS)

# NVHPC / CUDA library locations (adjust if your install path differs)
CUDA_LIB_DIR = $(NVHPC24_ROOT)/cuda/$(NVHPC_CUDA_VER)/lib64
NVHPC_MATH_LIB_DIR = $(NVHPC24_ROOT)/math_libs/$(NVHPC_CUDA_VER)/targets/x86_64-linux/lib
NVHPC_COMM_LIB_DIR = $(NVHPC24_ROOT)/comm_libs/$(NVHPC_CUDA_VER)/nccl/lib
NVHPC_COMPILER_LIB_DIR = $(NVHPC24_ROOT)/compilers/lib

OMPI_LIB_DIR = $(MPI_HOME)/lib
UCX_LIB_DIR = $(HPCX_ROOT)/ucx/lib
UCC_LIB_DIR = $(HPCX_ROOT)/ucc/lib

# rpath so runtime finds NVHPC + HPCX libs (even without LD_LIBRARY_PATH)
RPATH_FLAGS = -Wl,--disable-new-dtags \
	-Wl,-rpath,$(CUDA_LIB_DIR) \
	-Wl,-rpath,$(NVHPC_MATH_LIB_DIR) \
	-Wl,-rpath,$(NVHPC_COMM_LIB_DIR) \
	-Wl,-rpath,$(NVHPC_COMPILER_LIB_DIR) \
	-Wl,-rpath,$(OMPI_LIB_DIR) \
	-Wl,-rpath,$(UCX_LIB_DIR) \
	-Wl,-rpath,$(UCC_LIB_DIR)

LDFLAGS = -L$(CUDA_LIB_DIR) -L$(NVHPC_MATH_LIB_DIR) -L$(NVHPC_COMM_LIB_DIR) \
		 -L$(OMPI_LIB_DIR) -L$(UCX_LIB_DIR) -L$(UCC_LIB_DIR) \
		 $(RPATH_FLAGS) \
		 -lcusolver -lcusolverMp -lcublas -lcudart -lnccl -lcal

# Note: Linking to cusolverMp, NCCL and other NVHPC libs requires NVHPC / NVIDIA
# HPC SDK installed and visible at the paths above. Adjust these variables if
# you installed NVHPC/CUDA to different locations. See README for details.

# --- Build Rules ---

# The default goal: build both binaries.
all: $(EXE) $(TEST_EXE)

# Debug build target (no optimizations, include device debug info)
debug: NVCCFLAGS := -g -G -O0 --std=c++14 -DUSE_CUSOLVERMP $(ARCH) -Xcompiler -Wall -I$(CUDA_INC_DIR) -I$(NVHPC_INC_DIR) -I$(NCCL_INC_DIR) -I$(MPI_INC_DIR) $(MPI_CFLAGS_CLEAN)
debug: CXXFLAGS := -g -O0 --std=c++14 -DUSE_CUSOLVERMP -Wall -I$(CUDA_INC_DIR) -I$(NVHPC_INC_DIR) -I$(NCCL_INC_DIR) -I$(MPI_INC_DIR) $(MPI_CFLAGS)
debug: clean all


# Rule to link the final executable.
$(EXE): $(RBF_OBJECTS)
	@echo "==> Linking executable: $@"
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LDFLAGS) $(MPI_LDFLAGS)
	@echo "==> Build complete. Executable: ./$@"

$(TEST_EXE): $(TEST_OBJECTS)
	@echo "==> Linking executable: $@"
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LDFLAGS) $(MPI_LDFLAGS)
	@echo "==> Build complete. Executable: ./$@"

# Generic rule to compile .cu files into .o files.
%.o: %.cu
	@echo "==> Compiling CUDA source: $<"
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

# Generic rule to compile .cpp files into .o files.
# We use NVCC to compile .cpp files as well to ensure compatibility
# with CUDA headers, though g++ could also be used if there are no CUDA dependencies.
%.o: %.cpp
	@echo "==> Compiling C++ source: $<"
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

# Phony target for cleaning up the directory.
.PHONY: clean
clean:
	@echo "==> Cleaning up build files..."
	rm -f $(EXE) $(TEST_EXE) $(RBF_OBJECTS) $(TEST_OBJECTS)
	@echo "==> Done."
