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
		// Setup Delays
		// Get the randoms
		if (delay_range[0] == delay_range[1]) {
			delays[i] = delay_range[0];
		} else {
			float rnddelay = delay_range[0] + (delay_range[1] - delay_range[0])*((float)rand() / (RAND_MAX));
			delays[i] = round(rnddelay);
		}
		// Setup STDP
		if (stdp_on){
			stdp[i] = 1;
		} else {
			stdp[i] = 0;
		}
	}

}

void IzhikevichSpikingSynapses::increment_number_of_synapses(int increment) {

	SpikingSynapses::increment_number_of_synapses(increment);

}


void IzhikevichSpikingSynapses::allocate_device_pointers() {

	SpikingSynapses::allocate_device_pointers();

	reset_synapse_spikes();
}

void IzhikevichSpikingSynapses::reset_synapse_spikes() {

}


void IzhikevichSpikingSynapses::set_threads_per_block_and_blocks_per_grid(int threads) {
	
	SpikingSynapses::set_threads_per_block_and_blocks_per_grid(threads);
	
}


__global__ void izhikevich_calculate_postsynaptic_current_injection_kernal(float* d_synaptic_efficacies_or_weights,
							float* d_time_of_last_postsynaptic_activation_for_each_synapse,
							int* d_postsynaptic_neuron_indices,
							float* d_neurons_current_injections,
							float current_time_in_seconds,
							size_t total_number_of_synapses);

__global__ void izhikevich_apply_ltd_to_synapse_weights_kernal(float* d_time_of_last_postsynaptic_activation_for_each_synapse,
							float* d_synaptic_efficacies_or_weights,
							int* d_stdp,
							float* d_last_spike_time_of_each_neuron,
							int* d_postsyns,
							float currtime,
							struct stdp_struct stdp_vars,
							size_t total_number_of_synapse);

__global__ void izhikevich_apply_ltp_to_synapse_weights_kernal(int* d_postsyns,
							float* d_last_spike_time_of_each_neuron,
							int* d_stdp,
							float* d_time_of_last_postsynaptic_activation_for_each_synapse,
							float* d_synaptic_efficacies_or_weights,
							struct stdp_struct stdp_vars,
							float currtime,
							size_t total_number_of_synapse);



void IzhikevichSpikingSynapses::calculate_postsynaptic_current_injection(float* d_neurons_current_injections, float current_time_in_seconds) {

	izhikevich_calculate_postsynaptic_current_injection_kernal<<<number_of_synapse_blocks_per_grid, threads_per_block>>>(d_synaptic_efficacies_or_weights,
																	d_time_of_last_postsynaptic_activation_for_each_synapse,
																	d_postsynaptic_neuron_indices,
																	d_neurons_current_injections,
																	current_time_in_seconds,
																	total_number_of_synapses);

	CudaCheckError();
}

void IzhikevichSpikingSynapses::apply_ltd_to_synapse_weights(float* d_last_spike_time_of_each_neuron, float current_time_in_seconds) {

	izhikevich_apply_ltd_to_synapse_weights_kernal<<<number_of_synapse_blocks_per_grid, threads_per_block>>>(d_time_of_last_postsynaptic_activation_for_each_synapse,
																	d_synaptic_efficacies_or_weights,
																	d_stdp,
																	d_last_spike_time_of_each_neuron,
																	d_postsynaptic_neuron_indices,
																	current_time_in_seconds,
																	stdp_vars, // Should make device copy?
																	total_number_of_synapses);

	CudaCheckError();
}


void IzhikevichSpikingSynapses::apply_ltp_to_synapse_weights(float* d_last_spike_time_of_each_neuron, float current_time_in_seconds) {
	// Carry out the last step, LTP!
	izhikevich_apply_ltp_to_synapse_weights_kernal<<<number_of_synapse_blocks_per_grid, threads_per_block>>>(d_postsynaptic_neuron_indices,
																	d_last_spike_time_of_each_neuron,
																	d_stdp,
																	d_time_of_last_postsynaptic_activation_for_each_synapse,
																	d_synaptic_efficacies_or_weights,
																	stdp_vars, 
																	current_time_in_seconds,
																	total_number_of_synapses);

	CudaCheckError();
}


__global__ void izhikevich_calculate_postsynaptic_current_injection_kernal(float* d_synaptic_efficacies_or_weights,
							float* d_time_of_last_postsynaptic_activation_for_each_synapse,
							int* d_postsynaptic_neuron_indices,
							float* d_neurons_current_injections,
							float current_time_in_seconds,
							size_t total_number_of_synapses){

	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	if (idx < total_number_of_synapses) {

		if (d_time_of_last_postsynaptic_activation_for_each_synapse[idx] == current_time_in_seconds) {

			atomicAdd(&d_neurons_current_injections[d_postsynaptic_neuron_indices[idx]], d_synaptic_efficacies_or_weights[idx]);

		}
	}
	__syncthreads();
}


__global__ void izhikevich_apply_ltd_to_synapse_weights_kernal(float* d_time_of_last_postsynaptic_activation_for_each_synapse,
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
		if ((d_time_of_last_postsynaptic_activation_for_each_synapse[idx] == currtime) && (d_stdp[idx] == 1)) {
			float diff = d_last_spike_time_of_each_neuron[d_postsyns[idx]] - currtime;
			// STDP Update Rule
			float weightscale = stdp_vars.w_max * stdp_vars.a_minus * expf(diff / stdp_vars.tau_minus);
			// Now scale the weight (using an inverted column/row)
			d_synaptic_efficacies_or_weights[idx] += weightscale; 
		}
	}
}


// LTP on synapses
__global__ void izhikevich_apply_ltp_to_synapse_weights_kernal(int* d_postsyns,
							float* d_last_spike_time_of_each_neuron,
							int* d_stdp,
							float* d_time_of_last_postsynaptic_activation_for_each_synapse,
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
			float diff = currtime - d_time_of_last_postsynaptic_activation_for_each_synapse[idx];
			float weightchange = (stdp_vars.w_max - d_synaptic_efficacies_or_weights[idx]) * (stdp_vars.a_plus * expf(-diff / stdp_vars.tau_plus));
			// Update weights
			d_synaptic_efficacies_or_weights[idx] += weightchange;
		}

	}
}