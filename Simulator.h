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

#include "CUDAcode.h"
#include "Neurons.h"
#include "Connections.h"
// #include "Inputs.h"

// Simulator Class for running of the simulations
class Simulator{
public:
	// Constructor/Destructor
	Simulator();
	~Simulator();

	Neurons * neurons;
	Connections * connections;
	// Inputs * inputs;

	// Spike Generator related Data
	int numStimuli;
	int* numEntries;
	int** genids;
	float** gentimes;

	// Parameters
	float timestep;
	void SetTimestep(float timest);

	void SetNeuronType(Neurons * neurons_parameter);
	// void SetInputType(Inputs * inputs_parameter);

	int AddNeuronGroup(struct neuron_struct params, int shape[2]);
	// int AddInputGroup(input_struct group_params, int group_shape[2]);
	void AddConnectionGroup(int presynaptic_group_id, 
							int postsynaptic_group_id, 
							int connectivity_type,
							float weight_range[2], 
							float delay_range[2],
							bool stdp_on,
							float parameter = 0.0f,
							float parameter_two = 0.0f);


	void CreateGenerator(int popID, int stimulusid, int spikenumber, int* ids, float* spiketimes);

	void LoadWeights(int numWeights,
						float* newWeights);

	void Run(float total_time_per_epoch, int number_of_epochs, bool save_spikes = false, bool random_presentation = false);
};
#endif