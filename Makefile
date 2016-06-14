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
# CFLAGS += -lineinfo

# Mac OS X 10.9+ uses libc++, which is an implementation of c++11 standard library. 
# We must therefore specify c++11 as standard for out of the box compilation on Linux. 
UNAME_S := $(shell uname -s)
ifeq ($(UNAME_S),Linux)
	CFLAGS += --std=c++11 
endif





# Default
model: ${FILE}
directory: ${EXPERIMENT_DIRECTORY}


# Separating out the individual compilations so as not to compilation time
${FILE}: ObjectFiles/${FILE}.o ObjectFiles/Simulator.o ObjectFiles/Neurons.o ObjectFiles/SpikingNeurons.o ObjectFiles/IzhikevichSpikingNeurons.o ObjectFiles/LIFSpikingNeurons.o ObjectFiles/PoissonSpikingNeurons.o ObjectFiles/ImagePoissonSpikingNeurons.o ObjectFiles/FstreamWrapper.o ObjectFiles/GeneratorSpikingNeurons.o ObjectFiles/Synapses.o ObjectFiles/SpikingSynapses.o ObjectFiles/IzhikevichSpikingSynapses.o ObjectFiles/ConductanceSpikingSynapses.o ObjectFiles/RecordingElectrodes.o ObjectFiles/RandomStateManager.o ObjectFiles/SpikeAnalyser.o ObjectFiles/GraphPlotter.o ObjectFiles/TimerWithMessages.o
	$(CC) -lineinfo -lpython2.7 ObjectFiles/${FILE}.o ObjectFiles/Simulator.o ObjectFiles/Neurons.o ObjectFiles/SpikingNeurons.o ObjectFiles/IzhikevichSpikingNeurons.o ObjectFiles/LIFSpikingNeurons.o ObjectFiles/PoissonSpikingNeurons.o ObjectFiles/ImagePoissonSpikingNeurons.o ObjectFiles/FstreamWrapper.o ObjectFiles/GeneratorSpikingNeurons.o ObjectFiles/Synapses.o ObjectFiles/SpikingSynapses.o ObjectFiles/IzhikevichSpikingSynapses.o ObjectFiles/ConductanceSpikingSynapses.o ObjectFiles/RecordingElectrodes.o ObjectFiles/RandomStateManager.o ObjectFiles/SpikeAnalyser.o ObjectFiles/GraphPlotter.o ObjectFiles/TimerWithMessages.o -o ${EXPERIMENT_DIRECTORY}/bin/${FILE}


# Compiling the Model file
ObjectFiles/${FILE}.o: ${EXPERIMENT_DIRECTORY}/${FILE}.cpp
	$(CC) $(CFLAGS) ${EXPERIMENT_DIRECTORY}/${FILE}.cpp -o $@
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
# Compiling the LIFSpikingNeurons class
ObjectFiles/LIFSpikingNeurons.o: Neurons/LIFSpikingNeurons.cu
	$(CC) $(CFLAGS) Neurons/LIFSpikingNeurons.cu -o $@
# Compiling the PoissonSpikingNeurons class
ObjectFiles/PoissonSpikingNeurons.o: Neurons/PoissonSpikingNeurons.cu
	$(CC) $(CFLAGS) Neurons/PoissonSpikingNeurons.cu -o $@
# Compiling the ImagePoissonSpikingNeurons class
ObjectFiles/ImagePoissonSpikingNeurons.o: Neurons/ImagePoissonSpikingNeurons.cu
	$(CC) $(CFLAGS) Neurons/ImagePoissonSpikingNeurons.cu -o $@
# Compiling the FstreamWrapper class
ObjectFiles/FstreamWrapper.o: Helpers/FstreamWrapper.cpp
	$(CC) $(CFLAGS) Helpers/FstreamWrapper.cpp -o $@
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
# Compiling the ConductanceSpikingSynapses class
ObjectFiles/ConductanceSpikingSynapses.o: Synapses/ConductanceSpikingSynapses.cu
	$(CC) $(CFLAGS) Synapses/ConductanceSpikingSynapses.cu -o $@
# Compiling RecordingElectrodes class
ObjectFiles/RecordingElectrodes.o: RecordingElectrodes/RecordingElectrodes.cu
	$(CC) $(CFLAGS) RecordingElectrodes/RecordingElectrodes.cu -o $@
# Compiling RandomStateManager class
ObjectFiles/RandomStateManager.o: Helpers/RandomStateManager.cu
	$(CC) $(CFLAGS) Helpers/RandomStateManager.cu -o $@
# Compiling SpikeAnalyser class
ObjectFiles/SpikeAnalyser.o: SpikeAnalyser/SpikeAnalyser.cu
	$(CC) $(CFLAGS) SpikeAnalyser/SpikeAnalyser.cu -o $@
# Compiling GraphPlotter class
ObjectFiles/GraphPlotter.o: SpikeAnalyser/GraphPlotter.cu
	$(CC) $(CFLAGS) SpikeAnalyser/GraphPlotter.cu -lpython2.7 -o $@
# Compiling TimeWithMessages class
ObjectFiles/TimerWithMessages.o: Helpers/TimerWithMessages.cu
	$(CC) $(CFLAGS) Helpers/TimerWithMessages.cu -o $@

# Test script
test: Simulator.o Neurons.o SpikingNeurons.o IzhikevichSpikingNeurons.o PoissonSpikingNeurons.o ImagePoissonSpikingNeurons.o FstreamWrapper.o GeneratorSpikingNeurons.o Synapses.o RecordingElectrodes.o RandomStateManager.o SpikeAnalyser.o
	$(CC) Tests.cu Simulator.o Neurons.o SpikingNeurons.o IzhikevichSpikingNeurons.o PoissonSpikingNeurons.o ImagePoissonSpikingNeurons.o FstreamWrapper.o GeneratorSpikingNeurons.o Synapses.o RecordingElectrodes.o RandomStateManager.o SpikeAnalyser.o -o unittests
cleantest:
	rm *.o unittests

# Removing all created files
clean:
	rm ObjectFiles/*.o run
