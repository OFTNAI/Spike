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
${FILE}: ${FILE}.o Simulator.o ModelNeurons.o Connections.o NeuronPopulations.o CUDAcode.o NeuronDynamics.o STDPDynamics.o
	$(CC) ${FILE}.o Simulator.o ModelNeurons.o Connections.o NeuronPopulations.o CUDAcode.o NeuronDynamics.o STDPDynamics.o -o ${FILE}

# Compiling the Model file
${FILE}.o: ${FILE}.cpp
	$(CC) $(CFLAGS) ${FILE}.cpp
# Compiling the Simulator class
Simulator.o: Simulator.cpp
	$(CC) $(CFLAGS) Simulator.cpp
# Compiling the ModelNeurons class
ModelNeurons.o: ModelNeurons.cpp
	$(CC) $(CFLAGS) ModelNeurons.cpp
# Compiling the Connections class
Connections.o: Connections.cpp
	$(CC) $(CFLAGS) Connections.cpp
# Compiling the Neuron class
NeuronPopulations.o: NeuronPopulations.cpp
	$(CC) $(CFLAGS) NeuronPopulations.cpp
# # Compiling the Synapse class
# Synapse.o: Synapse.cpp
# 	$(CC) $(CFLAGS) Synapse.cpp
# Compiling the CUDA code
CUDAcode.o: CUDAcode.cu
	$(CC) $(CFLAGS) CUDAcode.cu
# Compiling the CUDA code
NeuronDynamics.o: NeuronDynamics.cu
	$(CC) $(CFLAGS) NeuronDynamics.cu
# Compiling the CUDA code
STDPDynamics.o: STDPDynamics.cu
	$(CC) $(CFLAGS) STDPDynamics.cu


# Test script
test: Simulator.o ModelNeurons.o Connections.o NeuronPopulations.o CUDAcode.o NeuronDynamics.o STDPDynamics.o
	$(CC) Tests.cu Simulator.o ModelNeurons.o Connections.o NeuronPopulations.o CUDAcode.o NeuronDynamics.o STDPDynamics.o -o unittests
cleantest:
	rm *.o unittests

# Removing all created files
clean:
	rm *.o run
