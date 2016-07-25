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
#include "../Neurons/InputSpikingNeurons.h"
#include "../Neurons/SpikingNeurons.h"
#include "../RecordingElectrodes/RecordingElectrodes.h"
#include "../SpikeAnalyser/SpikeAnalyser.h"
#include "../STDP/STDP.h"


// Simulator Class for running of the simulations
class Simulator{
public:
	// Constructor/Destructor
	Simulator();
	~Simulator();

	SpikingNeurons * neurons;
	SpikingSynapses * synapses;
	InputSpikingNeurons * input_neurons;
	STDP* stdp_rule; 

	RecordingElectrodes * recording_electrodes;
	RecordingElectrodes * input_recording_electrodes;

	// Flag: Enable for high accuracy spike storage, Disable for speed
	bool high_fidelity_spike_storage;

	// Parameters
	float timestep;
	void SetTimestep(float timest);

	void SetNeuronType(SpikingNeurons * neurons_parameter);
	void SetInputNeuronType(InputSpikingNeurons * neurons_parameter);
	void SetSynapseType(SpikingSynapses * synapses_parameter);
	void SetSTDPType(STDP* stdp_parameter);

	int AddNeuronGroup(neuron_parameters_struct * group_params);
	int AddInputNeuronGroup(neuron_parameters_struct * group_params);
	
	void AddSynapseGroup(int presynaptic_group_id, 
							int postsynaptic_group_id, 
							synapse_parameters_struct * synapse_params);

	void AddSynapseGroupsForNeuronGroupAndEachInputGroup(int postsynaptic_group_id, 
							synapse_parameters_struct * synapse_params);


	void LoadWeights(int numWeights, float* newWeights);

	void setup_network();
	void setup_recording_electrodes_for_neurons(int number_of_timesteps_per_device_spike_copy_check_param, int device_spike_store_size_multiple_of_total_neurons_param, float proportion_of_device_spike_store_full_before_copy_param);
	void setup_recording_electrodes_for_input_neurons(int number_of_timesteps_per_device_spike_copy_check_param, int device_spike_store_size_multiple_of_total_neurons_param, float proportion_of_device_spike_store_full_before_copy_param);

	void RunSimulation(float presentation_time_per_stimulus_per_epoch, int number_of_epochs, bool record_spikes, bool save_recorded_spikes_and_states_to_file, bool apply_stdp_to_relevant_synapses, bool count_spikes_per_neuron_for_single_cell_analysis, Stimuli_Presentation_Struct * stimuli_presentation_params, int stimulus_presentation_order_seed, SpikeAnalyser *spike_analyser,bool human_readable_storage);
	void RunSimulationToCountNeuronSpikes(float presentation_time_per_stimulus_per_epoch, bool record_spikes, bool save_recorded_spikes_and_states_to_file, SpikeAnalyser *spike_analyser, bool human_readable_storage, bool isTrained);
	void RunSimulationToTrainNetwork(float presentation_time_per_stimulus_per_epoch, int number_of_epochs, Stimuli_Presentation_Struct * stimuli_presentation_params, int stimulus_presentation_order_seed);

protected: 
	void per_timestep_instructions(float current_time_in_seconds, bool apply_stdp_to_relevant_synapses);
};
#endif
