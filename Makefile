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
${FILE}: ${FILE}.o Simulator.o Neurons.o SpikingNeurons.o IzhikevichSpikingNeurons.o PoissonSpikingNeurons.o GeneratorSpikingNeurons.o Synapses.o RecordingElectrodes.o
	$(CC) ${FILE}.o Simulator.o Neurons.o SpikingNeurons.o IzhikevichSpikingNeurons.o PoissonSpikingNeurons.o GeneratorSpikingNeurons.o Synapses.o RecordingElectrodes.o -o ${FILE}

# Compiling the Model file
${FILE}.o: ${FILE}.cpp
	$(CC) $(CFLAGS) ${FILE}.cpp
# Compiling the Simulator class
Simulator.o: Simulator.cu
	$(CC) $(CFLAGS) Simulator.cu
# Compiling the Neurons class
Neurons.o: Neurons/Neurons.cu
	$(CC) $(CFLAGS) Neurons/Neurons.cu
# Compiling the SpikingNeurons class
SpikingNeurons.o: Neurons/SpikingNeurons.cu
	$(CC) $(CFLAGS) Neurons/SpikingNeurons.cu
# Compiling the IzhikevichSpikingNeurons class
IzhikevichSpikingNeurons.o: Neurons/IzhikevichSpikingNeurons.cu
	$(CC) $(CFLAGS) Neurons/IzhikevichSpikingNeurons.cu
# Compiling the PoissonSpikingNeurons class
PoissonSpikingNeurons.o: Neurons/PoissonSpikingNeurons.cu
	$(CC) $(CFLAGS) Neurons/PoissonSpikingNeurons.cu
# Compiling the GeneratorSpikingNeurons class
GeneratorSpikingNeurons.o: Neurons/GeneratorSpikingNeurons.cu
	$(CC) $(CFLAGS) Neurons/GeneratorSpikingNeurons.cu
# Compiling the Synapses class
Synapses.o: Synapses/Synapses.cu
	$(CC) $(CFLAGS) Synapses/Synapses.cu
# Compiling RecordingElectrodes class
RecordingElectrodes.o: RecordingElectrodes.cu
	$(CC) $(CFLAGS) RecordingElectrodes.cu


# Test script
test: Simulator.o Neurons.o SpikingNeurons.o IzhikevichSpikingNeurons.o PoissonSpikingNeurons.o GeneratorSpikingNeurons.o Synapses.o RecordingElectrodes.o
	$(CC) Tests.cu Simulator.o Neurons.o SpikingNeurons.o IzhikevichSpikingNeurons.o PoissonSpikingNeurons.o GeneratorSpikingNeurons.o Synapses.o RecordingElectrodes.o -o unittests
cleantest:
	rm *.o unittests

# Removing all created files
clean:
	rm *.o run
