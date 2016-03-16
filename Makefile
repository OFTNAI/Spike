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



# Default
all: model

# Separating out the individual compilations so as not to compilation time
model: Model.o Spike.o NeuronPopulations.o Synapse.o CUDAcode.o NeuronDynamics.o
	$(CC) Model.o Spike.o NeuronPopulations.o Synapse.o CUDAcode.o NeuronDynamics.o -o run

# Compiling the Model file
Model.o: Model.cpp
	$(CC) $(CFLAGS) Model.cpp
# Compiling the Spike class
Spike.o: Spike.cpp
	$(CC) $(CFLAGS) Spike.cpp
# Compiling the Neuron class
NeuronPopulations.o: NeuronPopulations.cpp
	$(CC) $(CFLAGS) NeuronPopulations.cpp
# Compiling the Synapse class
Synapse.o: Synapse.cpp
	$(CC) $(CFLAGS) Synapse.cpp
# Compiling the CUDA code
CUDAcode.o: CUDAcode.cu
	$(CC) $(CFLAGS) CUDAcode.cu
# Compiling the CUDA code
NeuronDynamics.o: NeuronDynamics.cu
	$(CC) $(CFLAGS) NeuronDynamics.cu

# Test script
test: Spike.o NeuronPopulations.o Synapse.o CUDAcode.o NeuronDynamics.o
	$(CC) Tests.cu Spike.o NeuronPopulations.o Synapse.o CUDAcode.o NeuronDynamics.o -o unittests
cleantest:
	rm *.o unittests

# Removing all created files
clean:
	rm *.o run
