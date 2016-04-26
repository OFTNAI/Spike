#include "SpikingSynapses.h"

#include "../Helpers/CUDAErrorCheckHelpers.h"
#include "../Helpers/TerminalHelpers.h"

// SpikingSynapses Constructor
SpikingSynapses::SpikingSynapses() {

	delays = NULL;
	stdp = NULL;

	d_delays = NULL;
	d_spikes = NULL;
	d_stdp = NULL;
	d_lastactive = NULL;
	d_spikebuffer = NULL;
}

// SpikingSynapses Destructor
SpikingSynapses::~SpikingSynapses() {
	// Just need to free up the memory
	// Full Matrices
	free(delays);
	free(stdp);

	CudaSafeCall(cudaFree(d_delays));
	CudaSafeCall(cudaFree(d_spikes));
	CudaSafeCall(cudaFree(d_stdp));
	CudaSafeCall(cudaFree(d_lastactive));
	CudaSafeCall(cudaFree(d_spikebuffer));

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
void SpikingSynapses::AddGroup(int presynaptic_group_id, 
						int postsynaptic_group_id, 
						Neurons * neurons,
						Neurons * input_neurons,
						int connectivity_type,
						float weight_range[2],
						int delay_range[2],
						bool stdp_on,
						float parameter,
						float parameter_two) {
	
	
	Synapses::AddGroup(presynaptic_group_id, 
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

void SpikingSynapses::increment_number_of_synapses(int increment) {

	Synapses::increment_number_of_synapses(increment);

	printf("Increment2: %d\n", increment);

    delays = (int*)realloc(delays, total_number_of_synapses * sizeof(int));
    stdp = (int*)realloc(stdp, total_number_of_synapses * sizeof(int));

}


void SpikingSynapses::initialise_device_pointers() {

	Synapses::initialise_device_pointers();

	CudaSafeCall(cudaMalloc((void **)&d_delays, sizeof(int)*total_number_of_synapses));
	CudaSafeCall(cudaMalloc((void **)&d_spikes, sizeof(int)*total_number_of_synapses));
	CudaSafeCall(cudaMalloc((void **)&d_stdp, sizeof(int)*total_number_of_synapses));
	CudaSafeCall(cudaMalloc((void **)&d_lastactive, sizeof(float)*total_number_of_synapses));
	CudaSafeCall(cudaMalloc((void **)&d_spikebuffer, sizeof(int)*total_number_of_synapses));

	CudaSafeCall(cudaMemcpy(d_delays, delays, sizeof(int)*total_number_of_synapses, cudaMemcpyHostToDevice));
	CudaSafeCall(cudaMemcpy(d_stdp, stdp, sizeof(int)*total_number_of_synapses, cudaMemcpyHostToDevice));

	reset_synapse_spikes();
}

void SpikingSynapses::reset_synapse_spikes() {
	CudaSafeCall(cudaMemset(d_spikes, 0, sizeof(int)*total_number_of_synapses));
	CudaSafeCall(cudaMemset(d_lastactive, -1000.0f, sizeof(float)*total_number_of_synapses));
	CudaSafeCall(cudaMemset(d_spikebuffer, -1, sizeof(int)*total_number_of_synapses));
}


void SpikingSynapses::set_threads_per_block_and_blocks_per_grid(int threads) {
	
	Synapses::set_threads_per_block_and_blocks_per_grid(threads);
	
}



__global__ void calculate_postsynaptic_current_injection_for_synapse_kernal(int* d_spikes,
							float* d_weights,
							float* d_lastactive,
							int* d_postsynaptic_neuron_indices,
							float* d_neurons_current_injections,
							float current_time_in_seconds,
							size_t total_number_of_synapses);

__global__ void check_for_synapse_spike_arrival_kernal(int* d_presynaptic_neuron_indices,
								int* d_delays,
								int* d_spikes,
								float* d_neurons_last_spike_time,
								float* d_input_neurons_last_spike_time,
								int* d_spikebuffer,
								float currtime,
								size_t total_number_of_synapses);

__global__ void apply_ltd_to_synapse_weights_kernal(float* d_lastactive,
							float* d_weights,
							int* d_stdp,
							float* d_lastspiketime,
							int* d_postsyns,
							float currtime,
							struct stdp_struct stdp_vars,
							size_t numConns);

__global__ void apply_ltp_to_synapse_weights_kernal(int* d_postsyns,
							float* d_lastspiketime,
							int* d_stdp,
							float* d_lastactive,
							float* d_weights,
							struct stdp_struct stdp_vars,
							float currtime,
							size_t numConns);



void SpikingSynapses::calculate_postsynaptic_current_injection_for_synapse(float* d_neurons_current_injections, float current_time_in_seconds) {

	calculate_postsynaptic_current_injection_for_synapse_kernal<<<number_of_synapse_blocks_per_grid, threads_per_block>>>(d_spikes,
																	d_weights,
																	d_lastactive,
																	d_postsynaptic_neuron_indices,
																	d_neurons_current_injections,
																	current_time_in_seconds,
																	total_number_of_synapses);

	CudaCheckError();
}

void SpikingSynapses::check_for_synapse_spike_arrival(float* d_neurons_last_spike_time, float* d_input_neurons_last_spike_time, float current_time_in_seconds) {

	check_for_synapse_spike_arrival_kernal<<<number_of_synapse_blocks_per_grid, threads_per_block>>>(d_presynaptic_neuron_indices,
																		d_delays,
																		d_spikes,
																		d_neurons_last_spike_time,
																		d_input_neurons_last_spike_time,
																		d_spikebuffer,
																		current_time_in_seconds,
																		total_number_of_synapses);

	CudaCheckError();
}

void SpikingSynapses::apply_ltd_to_synapse_weights(float* d_lastspiketime, float current_time_in_seconds) {

	apply_ltd_to_synapse_weights_kernal<<<number_of_synapse_blocks_per_grid, threads_per_block>>>(d_lastactive,
																	d_weights,
																	d_stdp,
																	d_lastspiketime,
																	d_postsynaptic_neuron_indices,
																	current_time_in_seconds,
																	stdp_vars, // Should make device copy?
																	total_number_of_synapses);

	CudaCheckError();
}


void SpikingSynapses::apply_ltp_to_synapse_weights(float* d_lastspiketime, float current_time_in_seconds) {
	// Carry out the last step, LTP!
	apply_ltp_to_synapse_weights_kernal<<<number_of_synapse_blocks_per_grid, threads_per_block>>>(d_postsynaptic_neuron_indices,
																	d_lastspiketime,
																	d_stdp,
																	d_lastactive,
																	d_weights,
																	stdp_vars, 
																	current_time_in_seconds,
																	total_number_of_synapses);

	CudaCheckError();
}



// If spike has reached synapse add synapse weight to postsyn current injection
// Was currentcalc
__global__ void calculate_postsynaptic_current_injection_for_synapse_kernal(int* d_spikes,
							float* d_weights,
							float* d_lastactive,
							int* d_postsynaptic_neuron_indices,
							float* d_neurons_current_injections,
							float current_time_in_seconds,
							size_t total_number_of_synapses){

	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	if (idx < (total_number_of_synapses)) {
		// Decrememnt Spikes
		d_spikes[idx] -= 1;
		if (d_spikes[idx] == 0) {

			atomicAdd(&d_neurons_current_injections[d_postsynaptic_neuron_indices[idx]], d_weights[idx]);

			// Change lastactive
			d_lastactive[idx] = current_time_in_seconds;
			// Done!
		}
	}
	__syncthreads();
}


// Synapses carrying spikes
// JI GIVE A BETTER NAME
__global__ void check_for_synapse_spike_arrival_kernal(int* d_presynaptic_neuron_indices,
								int* d_delays,
								int* d_spikes,
								float* d_neurons_last_spike_time,
								float* d_input_neurons_last_spike_time,
								int* d_spikebuffer,
								float current_time_in_seconds,
								size_t total_number_of_synapses){
	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	if (idx < total_number_of_synapses) {
		// Reduce the spikebuffer by 1
		d_spikebuffer[idx] -= 1;

		int presynaptic_neuron_index = d_presynaptic_neuron_indices[idx];
		float presynaptic_neurons_last_spike_time;
		if (presynaptic_neuron_index < 0) {
			presynaptic_neurons_last_spike_time = d_input_neurons_last_spike_time[-1*presynaptic_neuron_index - 1];
		} else {
			presynaptic_neurons_last_spike_time = d_neurons_last_spike_time[presynaptic_neuron_index];
		}

		// Check if the neuron PRE has just fired and if the synapse exists
		if (presynaptic_neurons_last_spike_time == current_time_in_seconds){
			// Update the spikes with the correct delay
			if (d_spikes[idx] <= 0){
				d_spikes[idx] = d_delays[idx];
			} else if (d_spikebuffer[idx] <= 0){
				d_spikebuffer[idx] = d_delays[idx];
			}
		}
		// If there is no waiting spike
		if (d_spikes[idx] <= 0) {
			// Use the buffer if necessary
			if (d_spikebuffer[idx] > 0) {
				d_spikes[idx] = d_spikebuffer[idx];
			} else {
				d_spikes[idx] = -1;
				d_spikebuffer[idx] = -1;
			}
		}
		// If the buffer has a smaller time than the spike, switch them
		if ((d_spikebuffer[idx] > 0) && (d_spikebuffer[idx] < d_spikes[idx])){
			int temp = d_spikes[idx];
			d_spikes[idx] = d_spikebuffer[idx];
			d_spikebuffer[idx] = temp;
		}
	}
}



// LTD of weights
__global__ void apply_ltd_to_synapse_weights_kernal(float* d_lastactive,
							float* d_weights,
							int* d_stdp,
							float* d_lastspiketime,
							int* d_postsyns,
							float currtime,
							struct stdp_struct stdp_vars,
							size_t numConns){

	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	if (idx < (numConns)) {
		// Get the locations for updating
		// Get the synapses that are to be LTD'd
		if ((d_lastactive[idx] == currtime) && (d_stdp[idx] == 1)) {
			float diff = d_lastspiketime[d_postsyns[idx]] - currtime;
			// STDP Update Rule
			float weightscale = stdp_vars.w_max * stdp_vars.a_minus * expf(diff / stdp_vars.tau_minus);
			// Now scale the weight (using an inverted column/row)
			d_weights[idx] += weightscale; 
		}
	}
}


// LTP on synapses
__global__ void apply_ltp_to_synapse_weights_kernal(int* d_postsyns,
							float* d_lastspiketime,
							int* d_stdp,
							float* d_lastactive,
							float* d_weights,
							struct stdp_struct stdp_vars,
							float currtime,
							size_t numConns) {

	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	if (idx < numConns) {
		// Get the synapses upon which we should do LTP
		// Reversed indexing to check post->pre synapses
		if ((d_lastspiketime[d_postsyns[idx]] == currtime) && (d_stdp[idx] == 1)){
			// Get the last active time / weight of the synapse
			// Calc time difference and weight change
			float diff = currtime - d_lastactive[idx];
			float weightchange = (stdp_vars.w_max - d_weights[idx]) * (stdp_vars.a_plus * expf(-diff / stdp_vars.tau_plus));
			// Update weights
			d_weights[idx] += weightchange;
		}

	}
}