# SDV-BMT native measurement plane (Stage 1R-G).  Builds with the system C++ toolchain and the installed
# CUDA 13.0 toolkit (headers, NVRTC) and NVIDIA driver libraries.  Stage 2b-G real-model baseline adds cuDNN
# (approved 2026-10-08) for the NVIDIA runner only: the installed cuDNN 9 (pip package nvidia-cudnn-cu12).
#   make -C native            -> build/sdv-bmt-native-fake, build/sdv-bmt-native-nvidia, build/sdv-bmt-native-tests
#   make -C native portable   -> build/sdv-bmt-native-fake, build/sdv-bmt-native-tests (no CUDA / cuDNN needed)
#   make -C native raw-dump   -> build/sdv-bmt-raw-dump (offline validation tool, tools/stage2b_g_real/raw_dump.cpp)
# Every executable gets a build stamp <executable>.source-tree.sha256: the digest of the native source tree
# (python3 -m sdv_bmt.dut.native_source), plus the tool's own source for raw-dump.  A run refuses an executable
# whose stamp differs from the current sources (v0.1.2 safeguard), so every file of that digest is a
# prerequisite of every executable here.
CUDA_HOME ?= /usr/local/cuda-13.0
CUDNN_ROOT ?= /usr/local/lib/python3.10/dist-packages/nvidia/cudnn
CXX ?= g++
CXXFLAGS ?= -O2 -std=c++17 -Wall -Wextra -Werror
PYTHON ?= python3
BUILD := build
COMMON := $(wildcard src/common/*.hpp)
DIGEST_SRCS := Makefile $(wildcard src/*.cpp src/*/*.hpp tests/*.cpp)
RAW_DUMP_SRC := ../tools/stage2b_g_real/raw_dump.cpp
STAMP = PYTHONPATH=$(abspath ../src) $(PYTHON) -m sdv_bmt.dut.native_source $(abspath .)
NVIDIA_LINK = -L$(CUDA_HOME)/lib64 -Wl,-rpath,$(CUDA_HOME)/lib64 -lnvrtc -lcuda -lnvidia-ml -ldl \
	    -L$(CUDNN_ROOT)/lib -Wl,-rpath,$(CUDNN_ROOT)/lib -l:libcudnn.so.9

all: $(BUILD)/sdv-bmt-native-fake $(BUILD)/sdv-bmt-native-nvidia $(BUILD)/sdv-bmt-native-tests

portable: $(BUILD)/sdv-bmt-native-fake $(BUILD)/sdv-bmt-native-tests

raw-dump: $(BUILD)/sdv-bmt-raw-dump

$(BUILD):
	mkdir -p $(BUILD)

$(BUILD)/sdv-bmt-native-fake: $(DIGEST_SRCS) | $(BUILD)
	$(CXX) $(CXXFLAGS) -o $@ src/main_fake.cpp
	$(STAMP) > $@.source-tree.sha256

$(BUILD)/sdv-bmt-native-nvidia: $(DIGEST_SRCS) | $(BUILD)
	$(CXX) $(CXXFLAGS) -I$(CUDA_HOME)/include -I$(CUDNN_ROOT)/include -o $@ src/main_nvidia.cpp $(NVIDIA_LINK)
	$(STAMP) > $@.source-tree.sha256

$(BUILD)/sdv-bmt-native-tests: $(DIGEST_SRCS) | $(BUILD)
	$(CXX) $(CXXFLAGS) -Isrc -o $@ tests/test_main.cpp
	$(STAMP) > $@.source-tree.sha256

$(BUILD)/sdv-bmt-raw-dump: $(DIGEST_SRCS) $(RAW_DUMP_SRC) | $(BUILD)
	$(CXX) $(CXXFLAGS) -I$(CUDA_HOME)/include -I$(CUDNN_ROOT)/include -Isrc -o $@ $(RAW_DUMP_SRC) $(NVIDIA_LINK)
	$(STAMP) $(abspath $(RAW_DUMP_SRC)) > $@.source-tree.sha256

clean:
	rm -rf $(BUILD)

.PHONY: all portable raw-dump clean
