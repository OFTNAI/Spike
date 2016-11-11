//	Higgins STDP Class C++
//	HigginsSTDP.cu
//
//	Author: Nasir Ahmad
//	Date: 23/06/2016

#include "HigginsSTDP.h"
//CUDA #include "../Helpers/CUDAErrorCheckHelpers.h"
#include "../Helpers/TerminalHelpers.h"


// STDP Constructor
HigginsSTDP::HigginsSTDP() {

	stdp_params = NULL;
	syns = NULL;

}

// STDP Destructor
HigginsSTDP::~HigginsSTDP() {
	// free(stdp_params);
	// free(syns);
}

// Implementation of the STDP Rule for Irina's Model
void HigginsSTDP::Set_STDP_Parameters(SpikingSynapses* synapses, SpikingNeurons* neurons, SpikingNeurons* input_neurons, stdp_parameters_struct* stdp_parameters){
	stdp_params = (higgins_stdp_parameters_struct *)stdp_parameters;
	syns = synapses;
}

// Run the STDP
void HigginsSTDP::Run_STDP(float* d_last_spike_time_of_each_neuron, float current_time_in_seconds, float timestep){
	apply_ltd_to_synapse_weights(d_last_spike_time_of_each_neuron, current_time_in_seconds);
	apply_ltp_to_synapse_weights(d_last_spike_time_of_each_neuron, current_time_in_seconds);
}

void HigginsSTDP::apply_ltd_to_synapse_weights(float* d_last_spike_time_of_each_neuron, float current_time_in_seconds) {
  /*CUDA
	izhikevich_apply_ltd_to_synapse_weights_kernel<<<syns->number_of_synapse_blocks_per_grid, syns->threads_per_block>>>(
																	syns->d_time_of_last_spike_to_reach_synapse,
																	syns->d_synaptic_efficacies_or_weights,
																	syns->d_stdp,
																	d_last_spike_time_of_each_neuron,
																	syns->d_postsynaptic_neuron_indices,
																	current_time_in_seconds,
																	*stdp_params, // Should make device copy?
																	syns->total_number_of_synapses);

	CudaCheckError();
  */
}


void HigginsSTDP::apply_ltp_to_synapse_weights(float* d_last_spike_time_of_each_neuron, float current_time_in_seconds) {
  /*CUDA
	izhikevich_apply_ltp_to_synapse_weights_kernel<<<syns->number_of_synapse_blocks_per_grid, syns->threads_per_block>>>(
																	syns->d_postsynaptic_neuron_indices,
																	d_last_spike_time_of_each_neuron,
																	syns->d_stdp,
																	syns->d_time_of_last_spike_to_reach_synapse,
																	syns->d_synaptic_efficacies_or_weights,
																	*stdp_params, 
																	current_time_in_seconds,
																	syns->total_number_of_synapses);

	CudaCheckError();
  */
}


/*CUDA
// LTP on synapses
__global__ void izhikevich_apply_ltp_to_synapse_weights_kernel(int* d_postsyns,
							float* d_last_spike_time_of_each_neuron,
							bool* d_stdp,
							float* d_time_of_last_spike_to_reach_synapse,
							float* d_synaptic_efficacies_or_weights,
							struct higgins_stdp_parameters_struct stdp_vars,
							float currtime,
							size_t total_number_of_synapse) {

	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	while (idx < total_number_of_synapse) {
		// Get the synapses upon which we should do LTP
		// Reversed indexing to check post->pre synapses
		if ((d_last_spike_time_of_each_neuron[d_postsyns[idx]] == currtime) && (d_stdp[idx] == true)){
			// Get the last active time / weight of the synapse
			// Calc time difference and weight change
			float diff = currtime - d_time_of_last_spike_to_reach_synapse[idx];
			float weightchange = (stdp_vars.w_max - d_synaptic_efficacies_or_weights[idx]) * (stdp_vars.a_plus * expf(-diff / stdp_vars.tau_plus));
			// Update weights
			d_synaptic_efficacies_or_weights[idx] += weightchange;
		}
		idx += blockDim.x * gridDim.x;

	}
}


// LTD on Synapses
__global__ void izhikevich_apply_ltd_to_synapse_weights_kernel(float* d_time_of_last_spike_to_reach_synapse,
							float* d_synaptic_efficacies_or_weights,
							bool* d_stdp,
							float* d_last_spike_time_of_each_neuron,
							int* d_postsyns,
							float currtime,
							struct higgins_stdp_parameters_struct stdp_vars,
							size_t total_number_of_synapse){

	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	while (idx < total_number_of_synapse) {

		// Get the locations for updating
		// Get the synapses that are to be LTD'd
		if ((d_time_of_last_spike_to_reach_synapse[idx] == currtime) && (d_stdp[idx] == 1)) {
			float diff = d_last_spike_time_of_each_neuron[d_postsyns[idx]] - currtime;
			// STDP Update Rule
			float weightscale = stdp_vars.w_max * stdp_vars.a_minus * expf(diff / stdp_vars.tau_minus);
			// Now scale the weight (using an inverted column/row)
			d_synaptic_efficacies_or_weights[idx] += weightscale; 
		}
		idx += blockDim.x * gridDim.x;
	}
}
*/
