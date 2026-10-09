# SDV-BMT native measurement plane (Stage 1R-G).  Builds with the system C++ toolchain and the installed
# CUDA 13.0 toolkit (headers, NVRTC) and NVIDIA driver libraries.  Stage 2b-G real-model baseline adds cuDNN
# (approved 2026-10-08) for the NVIDIA runner only: the installed cuDNN 9 (pip package nvidia-cudnn-cu12).
#   make -C native            -> build/sdv-bmt-native-fake, build/sdv-bmt-native-nvidia, build/sdv-bmt-native-tests
#   make -C native portable   -> build/sdv-bmt-native-fake, build/sdv-bmt-native-tests (no CUDA / cuDNN needed)
CUDA_HOME ?= /usr/local/cuda-13.0
CUDNN_ROOT ?= /usr/local/lib/python3.10/dist-packages/nvidia/cudnn
CXX ?= g++
CXXFLAGS ?= -O2 -std=c++17 -Wall -Wextra -Werror
BUILD := build
COMMON := $(wildcard src/common/*.hpp)

all: $(BUILD)/sdv-bmt-native-fake $(BUILD)/sdv-bmt-native-nvidia $(BUILD)/sdv-bmt-native-tests

portable: $(BUILD)/sdv-bmt-native-fake $(BUILD)/sdv-bmt-native-tests

$(BUILD):
	mkdir -p $(BUILD)

$(BUILD)/sdv-bmt-native-fake: src/main_fake.cpp src/fake/backend.hpp $(COMMON) | $(BUILD)
	$(CXX) $(CXXFLAGS) -o $@ src/main_fake.cpp

$(BUILD)/sdv-bmt-native-nvidia: src/main_nvidia.cpp src/nvidia/backend.hpp src/nvidia/cudnn_plan.hpp $(COMMON) | $(BUILD)
	$(CXX) $(CXXFLAGS) -I$(CUDA_HOME)/include -I$(CUDNN_ROOT)/include -o $@ src/main_nvidia.cpp \
	    -L$(CUDA_HOME)/lib64 -Wl,-rpath,$(CUDA_HOME)/lib64 -lnvrtc -lcuda -lnvidia-ml -ldl \
	    -L$(CUDNN_ROOT)/lib -Wl,-rpath,$(CUDNN_ROOT)/lib -l:libcudnn.so.9

$(BUILD)/sdv-bmt-native-tests: tests/test_main.cpp src/fake/backend.hpp $(COMMON) | $(BUILD)
	$(CXX) $(CXXFLAGS) -Isrc -o $@ tests/test_main.cpp

clean:
	rm -rf $(BUILD)

.PHONY: all portable clean
