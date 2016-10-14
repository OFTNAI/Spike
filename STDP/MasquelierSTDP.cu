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
	neurs = neurons;
}

// 
void MasquelierSTDP::allocate_device_pointers(){
	// Add the correct space for d_temp_indices
	CudaSafeCall(cudaMalloc((void **)&d_temp_indices, sizeof(int)*syns->threads_per_block.x*neurs->total_number_of_neurons*2));
}

// Run the STDP
void MasquelierSTDP::Run_STDP(float* d_last_spike_time_of_each_neuron, float current_time_in_seconds, float timestep){
	apply_stdp_to_synapse_weights(d_last_spike_time_of_each_neuron, current_time_in_seconds);
}


void MasquelierSTDP::apply_stdp_to_synapse_weights(float* d_last_spike_time_of_each_neuron, float current_time_in_seconds) {
	// First reset the indices array
	// In order to carry out nearest spike potentiation only, we must find the spike arriving at each neuron which has the smallest time diff
	// I shall use a reduce algorithm for this purpose
	apply_stdp_to_synapse_weights_kernel<<<syns->number_of_synapse_blocks_per_grid, syns->threads_per_block>>>(
																	syns->d_postsynaptic_neuron_indices,
																	d_last_spike_time_of_each_neuron,
																	syns->d_stdp,
																	syns->d_time_of_last_spike_to_reach_synapse,
																	syns->d_synaptic_efficacies_or_weights,
																	d_temp_indices,
																	*stdp_params, 
																	current_time_in_seconds,
																	neurs->total_number_of_neurons,
																	syns->total_number_of_synapses);
	CudaCheckError();

	use_indices_to_apply_stdp<<<neurs->number_of_neuron_blocks_per_grid, syns->threads_per_block>>>(
																	syns->d_postsynaptic_neuron_indices,
																	d_last_spike_time_of_each_neuron,
																	syns->d_stdp,
																	syns->d_time_of_last_spike_to_reach_synapse,
																	syns->d_synaptic_efficacies_or_weights,
																	d_temp_indices,
																	*stdp_params, 
																	current_time_in_seconds,
																	syns->threads_per_block.x,
																	syns->total_number_of_synapses,
																	neurs->total_number_of_neurons);
	CudaCheckError();
}


// Find nearest spike
__global__ void apply_stdp_to_synapse_weights_kernel(int* d_postsyns,
							float* d_last_spike_time_of_each_neuron,
							bool* d_stdp,
							float* d_time_of_last_spike_to_reach_synapse,
							float* d_synaptic_efficacies_or_weights,
							int* indices,
							struct masquelier_stdp_parameters_struct stdp_vars,
							float currtime,
							int total_number_of_post_neurons,
							size_t total_number_of_synapse){
	// Global Index
	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	int tid = threadIdx.x;

	while (idx < total_number_of_post_neurons*blockDim.x*2){
			indices[idx] = -1000;
			idx += blockDim.x * gridDim.x;
	}

	idx = threadIdx.x + blockIdx.x * blockDim.x;
	

	// Running though all synapses
	while (idx < total_number_of_synapse) {
		if (d_stdp[idx] == true){
			// Get post-synaptic neuronid
			int n_idx = d_postsyns[idx];
			float last_syn_spike = d_time_of_last_spike_to_reach_synapse[idx];
			float last_post_spike = d_last_spike_time_of_each_neuron[d_postsyns[idx]];
			// If that synapse would be suitable for LTP
			if ((last_post_spike == currtime)){
				// Calculate time diff LTP
				float diff = currtime - last_syn_spike;

				float compdiff = 1000;
				if (indices[n_idx*blockDim.x + tid] >= 0)
					compdiff = currtime - d_time_of_last_spike_to_reach_synapse[indices[n_idx*blockDim.x + tid]];
				
				if (diff <= compdiff)
						indices[n_idx*blockDim.x + tid] = idx;
			}

			// If that synapse would be suitable for LTD
			if ((last_syn_spike == currtime)){
				// Calculate time diff LTD
				float diff = currtime - last_post_spike;

				float compdiff = 1000;
				if (indices[total_number_of_post_neurons*blockDim.x + n_idx*blockDim.x + tid] >= 0)
					compdiff = currtime - d_last_spike_time_of_each_neuron[
													d_postsyns[indices[total_number_of_post_neurons*blockDim.x + n_idx*blockDim.x + tid]]];
				
				if (diff <= compdiff)
						indices[total_number_of_post_neurons*blockDim.x + n_idx*blockDim.x + tid] = idx;
			}
		}

		idx += blockDim.x * gridDim.x;
	}
	__syncthreads();
}


__global__ void use_indices_to_apply_stdp(int* d_postsyns,
							float* d_last_spike_time_of_each_neuron,
							bool* d_stdp,
							float* d_time_of_last_spike_to_reach_synapse,
							float* d_synaptic_efficacies_or_weights,
							int* indices,
							struct masquelier_stdp_parameters_struct stdp_vars,
							float currtime,
							int synblock,
							int total_number_of_synapse,
							size_t total_number_of_post_neurons){
	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	int tid = threadIdx.x;

	// Running through the block now
	// Running through all neurons:
	while (idx < total_number_of_post_neurons){
		// Running through block
		for (unsigned int s=synblock/2; s>0; s>>=1){
			if (tid < s && idx < total_number_of_synapse){
				// Find MIN for LTP
				float diff1 = 1000;
				if (indices[idx*blockDim.x + tid] >= 0)
					diff1 = currtime - d_time_of_last_spike_to_reach_synapse[indices[idx*blockDim.x + tid]];
				float diff2 = 1000;
				if (indices[idx*blockDim.x + tid + s] >= 0)
					diff2 = currtime - d_time_of_last_spike_to_reach_synapse[indices[idx*blockDim.x + tid + s]];
				
				if (diff2 <= diff1){
					indices[idx*blockDim.x + tid] = indices[idx*blockDim.x + tid + s];
				}

				// Find MIN for LTD
				diff1 = 1000;
				if (indices[total_number_of_post_neurons*blockDim.x + idx*blockDim.x + tid] >= 0)
					diff1 = currtime - d_last_spike_time_of_each_neuron[
													d_postsyns[indices[total_number_of_post_neurons*blockDim.x + idx*blockDim.x + tid]]];
				diff2 = 1000;
				if (indices[total_number_of_post_neurons*blockDim.x + idx*blockDim.x + tid + s] >= 0)
					diff2 = currtime - d_last_spike_time_of_each_neuron[
													d_postsyns[indices[total_number_of_post_neurons*blockDim.x + idx*blockDim.x + tid + s]]];
				if (diff2 <= diff1){
					indices[total_number_of_post_neurons*blockDim.x + idx*blockDim.x + tid] = indices[total_number_of_post_neurons*blockDim.x + idx*blockDim.x + tid + s];
				}
			}
			__syncthreads();
		}
		idx += blockDim.x * gridDim.x;
	}

	// Reset Global Index
	idx = threadIdx.x + blockIdx.x * blockDim.x;


	while (idx < total_number_of_post_neurons){
		// Now get the zeroth element (i.e. the minimum)
		// If this neuron actually had some spike in the past LTP
		if ((indices[idx*synblock] >= 0) && (d_stdp[indices[idx*synblock]] == true)){
			float diff = currtime - d_time_of_last_spike_to_reach_synapse[indices[idx*synblock]];
			float new_syn_weight = d_synaptic_efficacies_or_weights[indices[idx*synblock]];
			// Only carry out LTP if the difference is in some range
			if (diff < 7*stdp_vars.tau_plus && diff > 0){
				float weightchange = stdp_vars.a_plus * expf(-diff / stdp_vars.tau_plus);
				// Update weights
				new_syn_weight += weightchange;
				// Ensure that the weights are clipped to 1.0f
				new_syn_weight = min(new_syn_weight, 1.0f);
				d_synaptic_efficacies_or_weights[indices[idx*synblock]] = new_syn_weight;
			}
		}
		// If this neuron actually had some spike in the past LTD
		if ((indices[total_number_of_post_neurons*synblock + idx*synblock] >= 0) && (d_stdp[indices[total_number_of_post_neurons*synblock + idx*synblock]] == true)){
			float diff = currtime - d_last_spike_time_of_each_neuron[
												d_postsyns[indices[total_number_of_post_neurons*synblock + idx*synblock]]];
			float new_syn_weight = d_synaptic_efficacies_or_weights[indices[total_number_of_post_neurons*synblock + idx*synblock]];
			// Only carry out LTP if the difference is in some range
			if (diff < 7*stdp_vars.tau_plus && diff > 0){
				float weightchange = stdp_vars.a_minus * expf(-diff / stdp_vars.tau_minus);
				// Update weights
				new_syn_weight -= weightchange;
				// Ensure that the weights are clipped to 1.0f
				new_syn_weight = max(new_syn_weight, 0.0f);
				d_synaptic_efficacies_or_weights[indices[total_number_of_post_neurons*synblock + idx*synblock]] = new_syn_weight;
			}
		}

		idx += blockDim.x * gridDim.x;
	}
}
