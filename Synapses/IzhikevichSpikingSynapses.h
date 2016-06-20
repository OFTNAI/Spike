#ifndef IZHIKEVICHSPIKINGSYNAPSES_H
#define IZHIKEVICHSPIKINGSYNAPSES_H

#include "SpikingSynapses.h"
#include "../Neurons/Neurons.h"


class IzhikevichSpikingSynapses : public SpikingSynapses {

public:

	// Constructor/Destructor
	IzhikevichSpikingSynapses();
	~IzhikevichSpikingSynapses();

	// Synapse Functions
	virtual void AddGroup(int presynaptic_group_id, 
						int postsynaptic_group_id, 
						Neurons * neurons,
						Neurons * input_neurons,
						float timestep,
						synapse_parameters_struct * synapse_params,
						float parameter,
						float parameter_two);

	virtual void allocate_device_pointers();
	virtual void reset_synapse_spikes();
	virtual void set_threads_per_block_and_blocks_per_grid(int threads);
	virtual void increment_number_of_synapses(int increment);
	virtual void shuffle_synapses();

	virtual void calculate_postsynaptic_current_injection(SpikingNeurons * neurons, float current_time_in_seconds);
	virtual void apply_ltd_to_synapse_weights(float* d_last_spike_time_of_each_neuron, float current_time_in_seconds);
	virtual void apply_ltp_to_synapse_weights(float* d_last_spike_time_of_each_neuron, float current_time_in_seconds);

};

__global__ void izhikevich_calculate_postsynaptic_current_injection_kernal(float* d_synaptic_efficacies_or_weights,
							float* d_time_of_last_spike_to_reach_synapse,
							int* d_postsynaptic_neuron_indices,
							float* d_neurons_current_injections,
							float current_time_in_seconds,
							size_t total_number_of_synapses);

__global__ void izhikevich_apply_ltd_to_synapse_weights_kernal(float* d_time_of_last_spike_to_reach_synapse,
							float* d_synaptic_efficacies_or_weights,
							bool* d_stdp,
							float* d_last_spike_time_of_each_neuron,
							int* d_postsyns,
							float currtime,
							struct stdp_struct stdp_vars,
							size_t total_number_of_synapse);

__global__ void izhikevich_apply_ltp_to_synapse_weights_kernal(int* d_postsyns,
							float* d_last_spike_time_of_each_neuron,
							bool* d_stdp,
							float* d_time_of_last_spike_to_reach_synapse,
							float* d_synaptic_efficacies_or_weights,
							struct stdp_struct stdp_vars,
							float currtime,
							size_t total_number_of_synapse);

#endif