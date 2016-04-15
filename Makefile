#--------------------------------------------------------------------------------
#
# Simple makefile for compiling the Spike Simulator
#
#--------------------------------------------------------------------------------

# Compiler = Nvidia version for CUDA code
CC = nvcc
# Flags
# -c = ignore the fact that there is no main
# --compiler-options -Wall = Warnings All. Give them to me.
# Wall flag is inefficient
CFLAGS = -c

# Mac OS X 10.9+ uses libc++, which is an implementation of c++11 standard library. 
# We must therefore specify c++11 as standard for out of the box compilation on Linux. 
UNAME_S := $(shell uname -s)
ifeq ($(UNAME_S),Linux)
	CFLAGS += --std=c++11 
endif


# Default
model: ${FILE}


# Separating out the individual compilations so as not to compilation time
${FILE}: ${FILE}.o Simulator.o Neurons.o IzhikevichNeurons.o Connections.o Inputs.o RecordingElectrodes.o CUDAcode.o
	$(CC) ${FILE}.o Simulator.o Neurons.o IzhikevichNeurons.o Connections.o Inputs.o RecordingElectrodes.o CUDAcode.o -o ${FILE}

# Compiling the Model file
${FILE}.o: ${FILE}.cpp
	$(CC) $(CFLAGS) ${FILE}.cpp
# Compiling the Simulator class
Simulator.o: Simulator.cpp
	$(CC) $(CFLAGS) Simulator.cpp
# Compiling the Neurons class
Neurons.o: Neurons.cu
	$(CC) $(CFLAGS) Neurons.cu
# Compiling the IzhikevichNeurons class
IzhikevichNeurons.o: IzhikevichNeurons.cu
	$(CC) $(CFLAGS) IzhikevichNeurons.cu
# Compiling the Connections class
Connections.o: Connections.cu
	$(CC) $(CFLAGS) Connections.cu
# Compiling the Inputs class
Inputs.o: Inputs.cu
	$(CC) $(CFLAGS) Inputs.cu
# Compiling RecordingElectrodes class
RecordingElectrodes.o: RecordingElectrodes.cu
	$(CC) $(CFLAGS) RecordingElectrodes.cu
# Compiling the CUDA code
CUDAcode.o: CUDAcode.cu
	$(CC) $(CFLAGS) CUDAcode.cu


# Test script
test: Simulator.o Neurons.o IzhikevichNeurons.o Connections.o Inputs.o RecordingElectrodes.o CUDAcode.o
	$(CC) Tests.cu Simulator.o Neurons.o IzhikevichNeurons.o Connections.o Inputs.o RecordingElectrodes.o CUDAcode.o -o unittests
cleantest:
	rm *.o unittests

# Removing all created files
clean:
	rm *.o run
