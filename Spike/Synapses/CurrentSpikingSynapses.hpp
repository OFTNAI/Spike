#ifndef CURRENTSPIKINGSYNAPSES_H
#define CURRENTSPIKINGSYNAPSES_H

#include "SpikingSynapses.hpp"
#include "../Neurons/Neurons.hpp"


class CurrentSpikingSynapses : public SpikingSynapses {

public:

	// Constructor/Destructor
	CurrentSpikingSynapses();
	~CurrentSpikingSynapses();


	virtual void AddGroup(int presynaptic_group_id, 
						int postsynaptic_group_id, 
						Neurons * neurons,
						Neurons * input_neurons,
						float timestep,
						synapse_parameters_struct * synapse_params);


	virtual void calculate_postsynaptic_current_injection(SpikingNeurons * neurons, float current_time_in_seconds, float timestep);
};

/*CUDA
__global__ void current_calculate_postsynaptic_current_injection_kernel(float* d_synaptic_efficacies_or_weights,
							float* d_time_of_last_spike_to_reach_synapse,
							int* d_postsynaptic_neuron_indices,
							float* d_neurons_current_injections,
							float current_time_in_seconds,
							size_t total_number_of_synapses);

__global__ void current_apply_ltd_to_synapse_weights_kernel(float* d_time_of_last_spike_to_reach_synapse,
							float* d_synaptic_efficacies_or_weights,
							bool* d_stdp,
							float* d_last_spike_time_of_each_neuron,
							int* d_postsyns,
							float currtime,
							struct stdp_struct stdp_vars,
							size_t total_number_of_synapse);

__global__ void current_apply_ltp_to_synapse_weights_kernel(int* d_postsyns,
							float* d_last_spike_time_of_each_neuron,
							bool* d_stdp,
							float* d_time_of_last_spike_to_reach_synapse,
							float* d_synaptic_efficacies_or_weights,
							struct stdp_struct stdp_vars,
							float currtime,
							size_t total_number_of_synapse);
*/

#endif
