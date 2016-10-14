//	Masquelier STDP Class C++
//	MasquelierSTDP.cu
//
//	Author: Nasir Ahmad
//	Date: 03/10/2016

#include "MasquelierSTDP.h"
#include "../Helpers/CUDAErrorCheckHelpers.h"
#include "../Helpers/TerminalHelpers.h"


// STDP Constructor
MasquelierSTDP::MasquelierSTDP() {

}

// STDP Destructor
MasquelierSTDP::~MasquelierSTDP() {

}

// Implementation of the STDP Rule for Irina's Model
void MasquelierSTDP::Set_STDP_Parameters(SpikingSynapses* synapses, SpikingNeurons* neurons, SpikingNeurons* input_neurons, stdp_parameters_struct* stdp_parameters){
	stdp_params = (masquelier_stdp_parameters_struct *)stdp_parameters;
	syns = synapses;
}

// Run the STDP
void MasquelierSTDP::Run_STDP(float* d_last_spike_time_of_each_neuron, float current_time_in_seconds, float timestep){
	apply_stdp_to_synapse_weights(d_last_spike_time_of_each_neuron, current_time_in_seconds);
}


void MasquelierSTDP::apply_stdp_to_synapse_weights(float* d_last_spike_time_of_each_neuron, float current_time_in_seconds) {

	apply_stdp_to_synapse_weights_kernel<<<syns->number_of_synapse_blocks_per_grid, syns->threads_per_block>>>(
																	syns->d_postsynaptic_neuron_indices,
																	d_last_spike_time_of_each_neuron,
																	syns->d_stdp,
																	syns->d_time_of_last_spike_to_reach_synapse,
																	syns->d_synaptic_efficacies_or_weights,
																	*stdp_params, 
																	current_time_in_seconds,
																	syns->total_number_of_synapses);
	CudaCheckError();
}


// STDP on synapses
__global__ void apply_stdp_to_synapse_weights_kernel(int* d_postsyns,
							float* d_last_spike_time_of_each_neuron,
							bool* d_stdp,
							float* d_time_of_last_spike_to_reach_synapse,
							float* d_synaptic_efficacies_or_weights,
							struct masquelier_stdp_parameters_struct stdp_vars,
							float currtime,
							size_t total_number_of_synapse) {

	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	while (idx < total_number_of_synapse) {
		if (d_stdp[idx] == true) {

			float last_post_spike = d_last_spike_time_of_each_neuron[d_postsyns[idx]];
			float last_syn_spike = d_time_of_last_spike_to_reach_synapse[idx];
			float new_syn_weight = d_synaptic_efficacies_or_weights[idx];

			// Get the synapses upon which we should do LTP
			// Reversed indexing to check post->pre synapses
			if ((last_post_spike == currtime)){
				// Get the last active time / weight of the synapse
				// Calc time difference and weight change
				float diff = currtime - last_syn_spike;
				// Only carry out LTP if the difference is in some range
				if (diff < 7*stdp_vars.tau_plus && diff > 0){
						float weightchange = stdp_vars.a_plus * expf(-diff / stdp_vars.tau_plus);
						// Update weights
						new_syn_weight += weightchange;
					// Ensure that the weights are clipped to 1.0f
					new_syn_weight = min(new_syn_weight, 1.0f);
				}
			}

			// Get the synapses upon which to do LTD
			if (last_syn_spike == currtime) {
				// STDP Update Rule
				float diff = currtime - last_post_spike;
				// Only carry out LTD if the difference is in some range
				if (diff < 7*stdp_vars.tau_minus && diff > 0){
					float weightchange = stdp_vars.a_minus * expf(-diff / stdp_vars.tau_minus);
					// Update the weights
					new_syn_weight -= weightchange;
					// Ensure that the weights are clipped to 0.0f
					new_syn_weight = max(new_syn_weight, 0.0f);
				}
			}

			d_synaptic_efficacies_or_weights[idx] = new_syn_weight;
		}
	
		idx += blockDim.x * gridDim.x;
	}
}