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

// cuRand Library import
#include <curand.h>
#include <curand_kernel.h>

//	CUDA library
#include <cuda.h>

// #include "CUDAcode.h"
#include "../Neurons/Neurons.h"
#include "../Synapses/Synapses.h"
#include "../Neurons/PoissonSpikingNeurons.h"
#include "../Neurons/SpikingNeurons.h"

// Simulator Class for running of the simulations
class Simulator{
public:
	// Constructor/Destructor
	Simulator();
	~Simulator();

	SpikingNeurons * neurons;
	Synapses * synapses;
	PoissonSpikingNeurons * input_neurons;

	// Spike Generator related Data
	int number_of_stimuli;
	int* numEntries;
	int** genids;
	float** gentimes;

	// Parameters
	float timestep;
	void SetTimestep(float timest);

	void SetNeuronType(SpikingNeurons * neurons_parameter);
	void SetInputNeuronType(PoissonSpikingNeurons * neurons_parameter);

	int AddNeuronGroup(neuron_parameters_struct * group_params, int shape[2]);
	int AddInputNeuronGroup(neuron_parameters_struct * group_params, int group_shape[2]);
	
	void AddSynapseGroup(int presynaptic_group_id, 
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

	void Run(float total_time_per_epoch, int number_of_epochs, bool save_spikes = false, bool present_stimuli_in_random_order = false);
};
#endif