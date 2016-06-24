#include "IzhikevichSpikingSynapses.h"

#include "../Helpers/CUDAErrorCheckHelpers.h"
#include "../Helpers/TerminalHelpers.h"

// IzhikevichSpikingSynapses Constructor
IzhikevichSpikingSynapses::IzhikevichSpikingSynapses() {

}

// IzhikevichSpikingSynapses Destructor
IzhikevichSpikingSynapses::~IzhikevichSpikingSynapses() {
	// Just need to free up the memory
}

// Connection Detail implementation
//	INPUT:
//		Pre-neuron population ID
//		Post-neuron population ID
//		An array of the exclusive sum of neuron populations
//		CONNECTIVITY_TYPE (Constants.h)
//		2 number float array for weight range
//		2 number float array for delay range
//		Boolean value to indicate if population is STDP based
//		Parameter = either probability for random synapses or S.D. for Gaussian
void IzhikevichSpikingSynapses::AddGroup(int presynaptic_group_id, 
						int postsynaptic_group_id, 
						Neurons * neurons,
						Neurons * input_neurons,
						float timestep,
						synapse_parameters_struct * synapse_params,
						float parameter,
						float parameter_two) {
	
	
	SpikingSynapses::AddGroup(presynaptic_group_id, 
							postsynaptic_group_id, 
							neurons,
							input_neurons,
							timestep,
							synapse_params,
							parameter,
							parameter_two);

}

void IzhikevichSpikingSynapses::increment_number_of_synapses(int increment) {
	SpikingSynapses::increment_number_of_synapses(increment);
}


void IzhikevichSpikingSynapses::allocate_device_pointers() {
	SpikingSynapses::allocate_device_pointers();
}

void IzhikevichSpikingSynapses::reset_synapse_spikes() {
	SpikingSynapses::reset_synapse_spikes();
}

void IzhikevichSpikingSynapses::shuffle_synapses() {
 	SpikingSynapses::shuffle_synapses();
}


void IzhikevichSpikingSynapses::set_threads_per_block_and_blocks_per_grid(int threads) {
	SpikingSynapses::set_threads_per_block_and_blocks_per_grid(threads);
}


void IzhikevichSpikingSynapses::calculate_postsynaptic_current_injection(SpikingNeurons * neurons, float current_time_in_seconds, float timestep) {

	izhikevich_calculate_postsynaptic_current_injection_kernel<<<number_of_synapse_blocks_per_grid, threads_per_block>>>(d_synaptic_efficacies_or_weights,
																	d_time_of_last_spike_to_reach_synapse,
																	d_postsynaptic_neuron_indices,
																	neurons->d_current_injections,
																	current_time_in_seconds,
																	total_number_of_synapses);

	CudaCheckError();
}


__global__ void izhikevich_calculate_postsynaptic_current_injection_kernel(float* d_synaptic_efficacies_or_weights,
							float* d_time_of_last_spike_to_reach_synapse,
							int* d_postsynaptic_neuron_indices,
							float* d_neurons_current_injections,
							float current_time_in_seconds,
							size_t total_number_of_synapses){

	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	if (idx < total_number_of_synapses) {

		if (d_time_of_last_spike_to_reach_synapse[idx] == current_time_in_seconds) {

			atomicAdd(&d_neurons_current_injections[d_postsynaptic_neuron_indices[idx]], d_synaptic_efficacies_or_weights[idx]);

		}
	}
	__syncthreads();
}