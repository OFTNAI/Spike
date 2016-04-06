// 	Simulator Class Header
// 	Simulator.h
//
//	Original Author: Nasir Ahmad
//	Date: 8/12/2015
//	Originally Spike.h
//  
//  Adapted by Nasir Ahmad and James Isbister
//	Date: 6/4/2016


#ifndef Simulator_H
#define Simulator_H
// Silences the printfs
//#define QUIETSTART

#include "Structs.h"
#include "NeuronPopulations.h"
// #include "Synapse.h"
#include "CUDAcode.h"
#include "ModelNeurons.h"
#include "Connections.h"

// Simulator Class for running of the simulations
class Simulator{
public:
	// Constructor/Destructor
	Simulator();
	~Simulator();
	
	NeuronPopulations population;
	// Synapse synconnects;

	ModelNeurons * model_neurons; // TO REPLACE NeuronPopulations EVENTUALLY
	Connections * connections;


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

	// JI
	void SetModelNeuronsObject(ModelNeurons * model_neurons_parameter);
	int AddModelNeuronsGroup(struct neuron_struct params, int shape[2]);
	void AddConnectionGroup(int presynaptic_group_id, 
							int postsynaptic_group_id, 
							int connectivity_type,
							float weight_range[2], 
							float delay_range[2],
							bool stdp_on,
							float parameter = 0.0f,
							float parameter_two = 0.0f);



	int CreateNeurons(int neuron_type, struct neuron_struct params, int shape[2]);
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