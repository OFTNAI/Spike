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
${FILE}: ObjectFiles/${FILE}.o ObjectFiles/Simulator.o ObjectFiles/Neurons.o ObjectFiles/SpikingNeurons.o ObjectFiles/IzhikevichSpikingNeurons.o ObjectFiles/PoissonSpikingNeurons.o ObjectFiles/GeneratorSpikingNeurons.o ObjectFiles/Synapses.o ObjectFiles/SpikingSynapses.o ObjectFiles/IzhikevichSpikingSynapses.o ObjectFiles/RecordingElectrodes.o
	$(CC) ObjectFiles/${FILE}.o ObjectFiles/Simulator.o ObjectFiles/Neurons.o ObjectFiles/SpikingNeurons.o ObjectFiles/IzhikevichSpikingNeurons.o ObjectFiles/PoissonSpikingNeurons.o ObjectFiles/GeneratorSpikingNeurons.o ObjectFiles/Synapses.o ObjectFiles/SpikingSynapses.o ObjectFiles/IzhikevichSpikingSynapses.o ObjectFiles/RecordingElectrodes.o -o ${FILE}

# Compiling the Model file
ObjectFiles/${FILE}.o: ${FILE}.cpp
	$(CC) $(CFLAGS) ${FILE}.cpp -o $@
# Compiling the Simulator class
ObjectFiles/Simulator.o: Simulator/Simulator.cu
	$(CC) $(CFLAGS) Simulator/Simulator.cu -o $@
# Compiling the Neurons class
ObjectFiles/Neurons.o: Neurons/Neurons.cu
	$(CC) $(CFLAGS) Neurons/Neurons.cu -o $@
# Compiling the SpikingNeurons class
ObjectFiles/SpikingNeurons.o: Neurons/SpikingNeurons.cu
	$(CC) $(CFLAGS) Neurons/SpikingNeurons.cu -o $@
# Compiling the IzhikevichSpikingNeurons class
ObjectFiles/IzhikevichSpikingNeurons.o: Neurons/IzhikevichSpikingNeurons.cu
	$(CC) $(CFLAGS) Neurons/IzhikevichSpikingNeurons.cu -o $@
# Compiling the PoissonSpikingNeurons class
ObjectFiles/PoissonSpikingNeurons.o: Neurons/PoissonSpikingNeurons.cu
	$(CC) $(CFLAGS) Neurons/PoissonSpikingNeurons.cu -o $@
# Compiling the GeneratorSpikingNeurons class
ObjectFiles/GeneratorSpikingNeurons.o: Neurons/GeneratorSpikingNeurons.cu
	$(CC) $(CFLAGS) Neurons/GeneratorSpikingNeurons.cu -o $@
# Compiling the Synapses class
ObjectFiles/Synapses.o: Synapses/Synapses.cu
	$(CC) $(CFLAGS) Synapses/Synapses.cu -o $@
# Compiling the SpikingSynapses class
ObjectFiles/SpikingSynapses.o: Synapses/SpikingSynapses.cu
	$(CC) $(CFLAGS) Synapses/SpikingSynapses.cu -o $@
# Compiling the IzhikevichSpikingSynapses class
ObjectFiles/IzhikevichSpikingSynapses.o: Synapses/IzhikevichSpikingSynapses.cu
	$(CC) $(CFLAGS) Synapses/IzhikevichSpikingSynapses.cu -o $@
# Compiling RecordingElectrodes class
ObjectFiles/RecordingElectrodes.o: RecordingElectrodes/RecordingElectrodes.cu
	$(CC) $(CFLAGS) RecordingElectrodes/RecordingElectrodes.cu -o $@


# Test script
test: Simulator.o Neurons.o SpikingNeurons.o IzhikevichSpikingNeurons.o PoissonSpikingNeurons.o GeneratorSpikingNeurons.o Synapses.o RecordingElectrodes.o
	$(CC) Tests.cu Simulator.o Neurons.o SpikingNeurons.o IzhikevichSpikingNeurons.o PoissonSpikingNeurons.o GeneratorSpikingNeurons.o Synapses.o RecordingElectrodes.o -o unittests
cleantest:
	rm *.o unittests

# Removing all created files
clean:
	rm ObjectFiles/*.o run
