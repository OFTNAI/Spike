// Spike Class Header
// Spike.h
//
//	Author: Nasir Ahmad
//	Date: 8/12/2015

#ifndef Spike_H
#define Spike_H
// Silences the printfs
//#define QUIETSTART

#include "Neuron.h"
#include "Synapse.h"
#include "CUDAcode.h"

// Spike Class for running of the simulations
class Spike{
public:
	// Constructor/Destructor
	Spike();
	~Spike();
	// Initialise Classes
	Neuron population;
	Synapse synconnects;
	// Poisson related data
	int numPoisson;
	int** poisson;
	float* poissonrate;
	float* poissonmask;
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
	float a_plus;
	float a_minus;
	float tau_plus;
	float tau_minus;
	float w_max;
	void SetTimestep(float timest);
	int CreateNeurons(int number, char type[], float params[]);
	void CreateGenerator(int popID, int stimulusid, int spikenumber, int* ids, float* spiketimes);
	void BeginConnections();
	void LoadWeights(int numWeights,
						float* newWeights);
	void CreateConnection(int pre, 
						int post, 
						char style[], 
						float weights[2], 
						float delays[2], 
						bool stdp,
						float parameter);
	void PoissonMask();
	void Run(float totaltime, int numEpochs, bool saveSpikes = false, bool randompresentation = false);
};
#endif