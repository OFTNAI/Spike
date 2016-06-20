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
#include "../Synapses/SpikingSynapses.h"
#include "../Neurons/PoissonSpikingNeurons.h"
#include "../Neurons/SpikingNeurons.h"
#include "../RecordingElectrodes/RecordingElectrodes.h"
#include "../SpikeAnalyser/SpikeAnalyser.h"


// Simulator Class for running of the simulations
class Simulator{
public:
	// Constructor/Destructor
	Simulator();
	~Simulator();

	SpikingNeurons * neurons;
	SpikingSynapses * synapses;
	PoissonSpikingNeurons * input_neurons;

	RecordingElectrodes * recording_electrodes;
	RecordingElectrodes * input_recording_electrodes;

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
	void SetSynapseType(SpikingSynapses * synapses_parameter);

	int AddNeuronGroup(neuron_parameters_struct * group_params);
	int AddInputNeuronGroup(neuron_parameters_struct * group_params);
	
	void AddSynapseGroup(int presynaptic_group_id, 
							int postsynaptic_group_id, 
							float delay_range[2],
							synapse_parameters_struct * synapse_params,
							float parameter = 0.0f,
							float parameter_two = 0.0f);

	void AddSynapseGroupsForNeuronGroupAndEachInputGroup(int postsynaptic_group_id, 
							float delay_range[2],
							synapse_parameters_struct * synapse_params,
							float parameter = 0.0f,
							float parameter_two = 0.0f);


	void CreateGenerator(int popID, int stimulusid, int spikenumber, int* ids, float* spiketimes);

	void LoadWeights(int numWeights, float* newWeights);

	void setup_network(bool temp_model_type);
	void setup_recording_electrodes_for_neurons(int number_of_timesteps_per_device_spike_copy_check_param, int device_spike_store_size_multiple_of_total_neurons_param, float proportion_of_device_spike_store_full_before_copy_param);
	void setup_recording_electrodes_for_input_neurons(int number_of_timesteps_per_device_spike_copy_check_param, int device_spike_store_size_multiple_of_total_neurons_param, float proportion_of_device_spike_store_full_before_copy_param);

	void RunSimulation(float presentation_time_per_stimulus_per_epoch, int number_of_epochs, int temp_model_type, bool record_spikes, bool save_recorded_spikes_to_file, bool apply_stdp_to_relevant_synapses, bool count_spikes_per_neuron_for_single_cell_analysis, bool present_stimuli_in_random_order, SpikeAnalyser *spike_analyser);
	void RunSimulationToCountNeuronSpikesForSingleCellAnalysis(float presentation_time_per_stimulus_per_epoch, int temp_model_type, bool record_spikes, bool save_recorded_spikes_to_file, SpikeAnalyser *spike_analyser);
	void RunSimulationToTrainNetwork(float presentation_time_per_stimulus_per_epoch, int temp_model_type, int number_of_epochs, bool present_stimuli_in_random_order);

protected: 
	void temp_izhikevich_per_timestep_instructions(float current_time_in_seconds);
	void temp_lif_per_timestep_instructions(float current_time_in_seconds, bool apply_stdp_to_relevant_synapses);

};
#endif