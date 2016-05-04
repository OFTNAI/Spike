#include "LIFSpikingSynapses.h"

#include "../Helpers/CUDAErrorCheckHelpers.h"
#include "../Helpers/TerminalHelpers.h"

// LIFSpikingSynapses Constructor
LIFSpikingSynapses::LIFSpikingSynapses() {
	synaptic_conductances_g = NULL;
	d_synaptic_conductances_g = NULL;

	recent_presynaptic_activities_C = NULL;
	d_recent_presynaptic_activities_C = NULL;
}

// LIFSpikingSynapses Destructor
LIFSpikingSynapses::~LIFSpikingSynapses() {
	free(synaptic_conductances_g);
	CudaSafeCall(cudaFree(d_synaptic_conductances_g));

	free(recent_presynaptic_activities_C);
	CudaSafeCall(cudaFree(d_recent_presynaptic_activities_C));
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
void LIFSpikingSynapses::AddGroup(int presynaptic_group_id, 
						int postsynaptic_group_id, 
						Neurons * neurons,
						Neurons * input_neurons,
						int connectivity_type,
						float weight_range[2],
						int delay_range[2],
						bool stdp_on,
						float parameter,
						float parameter_two) {
	
	
	SpikingSynapses::AddGroup(presynaptic_group_id, 
							postsynaptic_group_id, 
							neurons,
							input_neurons,
							connectivity_type, 
							weight_range,
							delay_range,
							stdp_on,
							parameter,
							parameter_two);

	for (int i = (total_number_of_synapses - temp_number_of_synapses_in_last_group); i < total_number_of_synapses-1; i++){
		synaptic_conductances_g[i] = 0.0f;
		recent_presynaptic_activities_C[i] = 0.0f;
	}

}

void LIFSpikingSynapses::increment_number_of_synapses(int increment) {

	SpikingSynapses::increment_number_of_synapses(increment);

	synaptic_conductances_g = (float*)realloc(synaptic_conductances_g, total_number_of_synapses * sizeof(float));
	recent_presynaptic_activities_C = (float*)realloc(recent_presynaptic_activities_C, total_number_of_synapses * sizeof(float));

}


void LIFSpikingSynapses::allocate_device_pointers() {

	SpikingSynapses::allocate_device_pointers();

	CudaSafeCall(cudaMalloc((void **)&d_synaptic_conductances_g, sizeof(float)*total_number_of_synapses));
	CudaSafeCall(cudaMalloc((void **)&d_recent_presynaptic_activities_C, sizeof(float)*total_number_of_synapses));

}

void LIFSpikingSynapses::reset_synapse_spikes() {

	SpikingSynapses::reset_synapse_spikes();

	// CudaSafeCall(cudaMemset(d_synaptic_conductances_g, 0.0f, sizeof(float)*total_number_of_synapses));
	CudaSafeCall(cudaMemcpy(d_synaptic_conductances_g, synaptic_conductances_g, sizeof(float)*total_number_of_synapses, cudaMemcpyHostToDevice));
	CudaSafeCall(cudaMemcpy(d_recent_presynaptic_activities_C, recent_presynaptic_activities_C, sizeof(float)*total_number_of_synapses, cudaMemcpyHostToDevice));

}


void LIFSpikingSynapses::shuffle_synapses() {
	SpikingSynapses::shuffle_synapses();
}


void LIFSpikingSynapses::set_threads_per_block_and_blocks_per_grid(int threads) {
	
	SpikingSynapses::set_threads_per_block_and_blocks_per_grid(threads);
	
}


__global__ void lif_calculate_postsynaptic_current_injection_kernal(float* d_synaptic_efficacies_or_weights,
							float* d_time_of_last_spike_to_reach_synapse,
							int* d_postsynaptic_neuron_indices,
							float* d_neurons_current_injections,
							float current_time_in_seconds,
							size_t total_number_of_synapses,
							float * d_membrane_potentials_v,
							float * d_synaptic_conductances_g);


__global__ void lif_update_synaptic_conductances_kernal(float timestep, 
													float * d_synaptic_conductances_g, 
													float * d_synaptic_efficacies_or_weights, 
													float * d_time_of_last_spike_to_reach_synapse,
													int total_number_of_synapses,
													float current_time_in_seconds);


__global__ void lif_update_presynaptic_activities_C_kernal(float* d_recent_presynaptic_activities_C,
							float* d_time_of_last_spike_to_reach_synapse,
							int* d_stdp,
							float timestep,
							float current_time_in_seconds,
							size_t total_number_of_synapses);



//OLD
__global__ void lif_apply_ltd_to_synapse_weights_kernal(float* d_time_of_last_spike_to_reach_synapse,
							float* d_synaptic_efficacies_or_weights,
							int* d_stdp,
							float* d_last_spike_time_of_each_neuron,
							int* d_postsyns,
							float currtime,
							struct stdp_struct stdp_vars,
							size_t total_number_of_synapse);

__global__ void lif_apply_ltp_to_synapse_weights_kernal(int* d_postsyns,
							float* d_last_spike_time_of_each_neuron,
							int* d_stdp,
							float* d_time_of_last_spike_to_reach_synapse,
							float* d_synaptic_efficacies_or_weights,
							struct stdp_struct stdp_vars,
							float currtime,
							size_t total_number_of_synapse);


void LIFSpikingSynapses::calculate_postsynaptic_current_injection(SpikingNeurons * neurons, float current_time_in_seconds) {

	lif_calculate_postsynaptic_current_injection_kernal<<<number_of_synapse_blocks_per_grid, threads_per_block>>>(d_synaptic_efficacies_or_weights,
																	d_time_of_last_spike_to_reach_synapse,
																	d_postsynaptic_neuron_indices,
																	neurons->d_current_injections,
																	current_time_in_seconds,
																	total_number_of_synapses,
																	neurons->d_membrane_potentials_v, 
																	d_synaptic_conductances_g);

	CudaCheckError();
}

void LIFSpikingSynapses::update_synaptic_conductances(float timestep, float current_time_in_seconds) {

	lif_update_synaptic_conductances_kernal<<<number_of_synapse_blocks_per_grid, threads_per_block>>>(timestep, 
											d_synaptic_conductances_g, 
											d_synaptic_efficacies_or_weights, 
											d_time_of_last_spike_to_reach_synapse,
											total_number_of_synapses,
											current_time_in_seconds);

	CudaCheckError();

}

void LIFSpikingSynapses::update_presynaptic_activities(float timestep, float current_time_in_seconds) {
	
	lif_update_presynaptic_activities_C_kernal<<<number_of_synapse_blocks_per_grid, threads_per_block>>>(d_recent_presynaptic_activities_C,
							d_time_of_last_spike_to_reach_synapse,
							d_stdp,
							timestep,
							current_time_in_seconds,
							total_number_of_synapses);

	CudaCheckError();
}


void LIFSpikingSynapses::apply_ltd_to_synapse_weights(float* d_last_spike_time_of_each_neuron, float current_time_in_seconds) {

	lif_apply_ltd_to_synapse_weights_kernal<<<number_of_synapse_blocks_per_grid, threads_per_block>>>(d_time_of_last_spike_to_reach_synapse,
																	d_synaptic_efficacies_or_weights,
																	d_stdp,
																	d_last_spike_time_of_each_neuron,
																	d_postsynaptic_neuron_indices,
																	current_time_in_seconds,
																	stdp_vars, // Should make device copy?
																	total_number_of_synapses);

	CudaCheckError();
}


void LIFSpikingSynapses::apply_ltp_to_synapse_weights(float* d_last_spike_time_of_each_neuron, float current_time_in_seconds) {
	// Carry out the last step, LTP!
	lif_apply_ltp_to_synapse_weights_kernal<<<number_of_synapse_blocks_per_grid, threads_per_block>>>(d_postsynaptic_neuron_indices,
																	d_last_spike_time_of_each_neuron,
																	d_stdp,
																	d_time_of_last_spike_to_reach_synapse,
																	d_synaptic_efficacies_or_weights,
																	stdp_vars, 
																	current_time_in_seconds,
																	total_number_of_synapses);

	CudaCheckError();
}



__global__ void lif_calculate_postsynaptic_current_injection_kernal(float* d_synaptic_efficacies_or_weights,
							float* d_time_of_last_spike_to_reach_synapse,
							int* d_postsynaptic_neuron_indices,
							float* d_neurons_current_injections,
							float current_time_in_seconds,
							size_t total_number_of_synapses,
							float * d_membrane_potentials_v,
							float * d_synaptic_conductances_g){

	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	if (idx < total_number_of_synapses) {

		float temp_reversal_potential_Vhat = 0.0;
		float membrane_potential_v = d_membrane_potentials_v[d_postsynaptic_neuron_indices[idx]];
		float synaptic_conductance_g = d_synaptic_conductances_g[idx];

		float component_for_sum = synaptic_conductance_g * (temp_reversal_potential_Vhat - membrane_potential_v);

		atomicAdd(&d_neurons_current_injections[d_postsynaptic_neuron_indices[idx]], component_for_sum);

	}
	__syncthreads();
}



__global__ void lif_update_synaptic_conductances_kernal(float timestep, 
														float * d_synaptic_conductances_g, 
														float * d_synaptic_efficacies_or_weights, 
														float * d_time_of_last_spike_to_reach_synapse, 
														int total_number_of_synapses,
														float current_time_in_seconds) {

	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	if (idx < total_number_of_synapses) {

		float synaptic_conductance_g = d_synaptic_conductances_g[idx];
		float decay_term_tau_g = 0.01; // Is this the synaptic time constant?

		float new_conductance = (1 - (timestep/decay_term_tau_g)) * synaptic_conductance_g;
		if (d_time_of_last_spike_to_reach_synapse[idx] == current_time_in_seconds) {
			new_conductance += timestep * d_synaptic_efficacies_or_weights[idx];
		}

		d_synaptic_conductances_g[idx] = new_conductance;

	}

}


__global__ void lif_update_presynaptic_activities_C_kernal(float* d_recent_presynaptic_activities_C,
							float* d_time_of_last_spike_to_reach_synapse,
							int* d_stdp,
							float timestep, 
							float current_time_in_seconds,
							size_t total_number_of_synapses) {

	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	if (idx < total_number_of_synapses) {

		if (d_stdp[idx] == 1) {

			float recent_presynaptic_activity_C = d_recent_presynaptic_activities_C[idx];
			float decay_term_tau_C = 0.01; // Should be variable between 0.003 and 0.075

			float new_recent_presynaptic_activity_C = (1 - (timestep/decay_term_tau_C)) * recent_presynaptic_activity_C;
			if (d_time_of_last_spike_to_reach_synapse[idx] == current_time_in_seconds) {
				float model_parameter_alpha_c = 0.5;
				new_recent_presynaptic_activity_C += timestep * model_parameter_alpha_c * (1 - recent_presynaptic_activity_C);
			}

			d_recent_presynaptic_activities_C[idx] = new_recent_presynaptic_activity_C;

		}

	}


}




__global__ void lif_apply_ltd_to_synapse_weights_kernal(float* d_time_of_last_spike_to_reach_synapse,
							float* d_synaptic_efficacies_or_weights,
							int* d_stdp,
							float* d_last_spike_time_of_each_neuron,
							int* d_postsyns,
							float currtime,
							struct stdp_struct stdp_vars,
							size_t total_number_of_synapse){

	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	if (idx < total_number_of_synapse) {

		// Get the locations for updating
		// Get the synapses that are to be LTD'd
		if ((d_time_of_last_spike_to_reach_synapse[idx] == currtime) && (d_stdp[idx] == 1)) {
			float diff = d_last_spike_time_of_each_neuron[d_postsyns[idx]] - currtime;
			// STDP Update Rule
			float weightscale = stdp_vars.w_max * stdp_vars.a_minus * expf(diff / stdp_vars.tau_minus);
			// Now scale the weight (using an inverted column/row)
			d_synaptic_efficacies_or_weights[idx] += weightscale; 
		}
	}
}


// LTP on synapses
__global__ void lif_apply_ltp_to_synapse_weights_kernal(int* d_postsyns,
							float* d_last_spike_time_of_each_neuron,
							int* d_stdp,
							float* d_time_of_last_spike_to_reach_synapse,
							float* d_synaptic_efficacies_or_weights,
							struct stdp_struct stdp_vars,
							float currtime,
							size_t total_number_of_synapse) {

	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	if (idx < total_number_of_synapse) {
		// Get the synapses upon which we should do LTP
		// Reversed indexing to check post->pre synapses
		if ((d_last_spike_time_of_each_neuron[d_postsyns[idx]] == currtime) && (d_stdp[idx] == 1)){
			// Get the last active time / weight of the synapse
			// Calc time difference and weight change
			float diff = currtime - d_time_of_last_spike_to_reach_synapse[idx];
			float weightchange = (stdp_vars.w_max - d_synaptic_efficacies_or_weights[idx]) * (stdp_vars.a_plus * expf(-diff / stdp_vars.tau_plus));
			// Update weights
			d_synaptic_efficacies_or_weights[idx] += weightchange;
		}

	}
}