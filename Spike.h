// Spike Class Header
// Spike.h
//
//	Author: Nasir Ahmad
//	Date: 8/12/2015

#ifndef Spike_H
#define Spike_H
// Silences the printfs
//#define QUIETSTART

#include "Structs.h"
#include "NeuronPopulations.h"
#include "Synapse.h"
#include "CUDAcode.h"

// Spike Class for running of the simulations
class Spike{
public:
	// Constructor/Destructor
	Spike();
	~Spike();
	// Initialise Classes
	NeuronPopulations population;
	Synapse synconnects;
	// Spike Generator related Data
	int numStimuli;
	int* numEntries;
	int** genids;
	float** gentimes;
	// Numbers of Neurons/Synapses
	int numPops;
	int numConnects;
	// Parameters
	float timestep;
	void SetTimestep(float timest);
	int CreateNeurons(int number, int neuron_type, struct neuron_struct params, int shape[2]);
	void CreateGenerator(int popID, int stimulusid, int spikenumber, int* ids, float* spiketimes);
	void BeginConnections();
	void LoadWeights(int numWeights,
						float* newWeights);
	void CreateConnection(int prepop, 
						int postpop, 
						int connectivity_type,
						float weights[2], 
						float delays[2], 
						bool stdp,
						float parameter = 0.0f,
						float parameter_two = 0.0f);
	void Run(float totaltime, int numEpochs, bool saveSpikes = false, bool randomPresentation = false);
};
#endif