#include "SpikingSynapses.h"

#include "../Helpers/CUDAErrorCheckHelpers.h"
#include "../Helpers/TerminalHelpers.h"

// SpikingSynapses Constructor
SpikingSynapses::SpikingSynapses() {

	delays = NULL;
	stdp = NULL;

	d_delays = NULL;
	d_spikes_travelling_to_synapse = NULL;
	d_stdp = NULL;
	d_time_of_last_spike_to_reach_synapse = NULL;
	d_spikes_travelling_to_synapse_buffer = NULL;
}

// SpikingSynapses Destructor
SpikingSynapses::~SpikingSynapses() {
	// Just need to free up the memory
	// Full Matrices
	free(delays);
	free(stdp);

	CudaSafeCall(cudaFree(d_delays));
	CudaSafeCall(cudaFree(d_spikes_travelling_to_synapse));
	CudaSafeCall(cudaFree(d_stdp));
	CudaSafeCall(cudaFree(d_time_of_last_spike_to_reach_synapse));
	CudaSafeCall(cudaFree(d_spikes_travelling_to_synapse_buffer));

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
						connectivity_parameters_struct * connectivity_params,
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
							connectivity_params,
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
		stdp[i] = stdp_on;
	}

}

void SpikingSynapses::increment_number_of_synapses(int increment) {

	Synapses::increment_number_of_synapses(increment);

    delays = (int*)realloc(delays, total_number_of_synapses * sizeof(int));
    stdp = (bool*)realloc(stdp, total_number_of_synapses * sizeof(bool));

}


void SpikingSynapses::allocate_device_pointers() {

	Synapses::allocate_device_pointers();

	CudaSafeCall(cudaMalloc((void **)&d_delays, sizeof(int)*total_number_of_synapses));
	CudaSafeCall(cudaMalloc((void **)&d_spikes_travelling_to_synapse, sizeof(int)*total_number_of_synapses));
	CudaSafeCall(cudaMalloc((void **)&d_stdp, sizeof(bool)*total_number_of_synapses));
	CudaSafeCall(cudaMalloc((void **)&d_time_of_last_spike_to_reach_synapse, sizeof(float)*total_number_of_synapses));
	CudaSafeCall(cudaMalloc((void **)&d_spikes_travelling_to_synapse_buffer, sizeof(int)*total_number_of_synapses));

	CudaSafeCall(cudaMemcpy(d_delays, delays, sizeof(int)*total_number_of_synapses, cudaMemcpyHostToDevice));
	CudaSafeCall(cudaMemcpy(d_stdp, stdp, sizeof(bool)*total_number_of_synapses, cudaMemcpyHostToDevice));

}

void SpikingSynapses::reset_synapse_spikes() {
	
	CudaSafeCall(cudaMemset(d_spikes_travelling_to_synapse, 0, sizeof(int)*total_number_of_synapses));
	CudaSafeCall(cudaMemset(d_time_of_last_spike_to_reach_synapse, -1000.0f, sizeof(float)*total_number_of_synapses));
	CudaSafeCall(cudaMemset(d_spikes_travelling_to_synapse_buffer, -1, sizeof(int)*total_number_of_synapses));
}


void SpikingSynapses::shuffle_synapses() {
	
	Synapses::shuffle_synapses();

	int * temp_delays = (int *)malloc(total_number_of_synapses*sizeof(int));
	bool * temp_stdp = (bool *)malloc(total_number_of_synapses*sizeof(bool));
	for(int i = 0; i < total_number_of_synapses; i++) {

		temp_delays[i] = delays[original_synapse_indices[i]];
		temp_stdp[i] = stdp[original_synapse_indices[i]];

	}

	delays = temp_delays;
	stdp = temp_stdp;

}


void SpikingSynapses::set_threads_per_block_and_blocks_per_grid(int threads) {
	
	Synapses::set_threads_per_block_and_blocks_per_grid(threads);
	
}



void SpikingSynapses::check_for_synapse_spike_arrival(float current_time_in_seconds) {

	// printf("check_for_synapse_spike_arrival. number_of_synapse_blocks_per_grid.x: %d. threads_per_block.x: %d\n", number_of_synapse_blocks_per_grid.x, threads_per_block.x);

	check_for_synapse_spike_arrival_kernal<<<number_of_synapse_blocks_per_grid, threads_per_block>>>(d_spikes_travelling_to_synapse,
																	d_time_of_last_spike_to_reach_synapse,
																	current_time_in_seconds,
																	total_number_of_synapses);

	CudaCheckError();
}

void SpikingSynapses::calculate_postsynaptic_current_injection(SpikingNeurons * neurons, float current_time_in_seconds) {

}

void SpikingSynapses::update_presynaptic_activities(float timestep, float current_time_in_seconds) {

}

void SpikingSynapses::update_synaptic_efficacies_or_weights(float * d_recent_postsynaptic_activities_D, float timestep, float current_time_in_seconds, float * d_last_spike_time_of_each_neuron) {
	
}


void SpikingSynapses::move_spikes_towards_synapses(float* d_last_spike_time_of_each_neuron, float* d_input_neurons_last_spike_time, float current_time_in_seconds) {

	move_spikes_towards_synapses_kernal<<<number_of_synapse_blocks_per_grid, threads_per_block>>>(d_presynaptic_neuron_indices,
																		d_delays,
																		d_spikes_travelling_to_synapse,
																		d_last_spike_time_of_each_neuron,
																		d_input_neurons_last_spike_time,
																		d_spikes_travelling_to_synapse_buffer,
																		current_time_in_seconds,
																		total_number_of_synapses);

	CudaCheckError();
}

void SpikingSynapses::apply_ltd_to_synapse_weights(float* d_last_spike_time_of_each_neuron, float current_time_in_seconds) {

}


void SpikingSynapses::apply_ltp_to_synapse_weights(float* d_last_spike_time_of_each_neuron, float current_time_in_seconds) {

}


__global__ void check_for_synapse_spike_arrival_kernal(int* d_spikes_travelling_to_synapse,
							float* d_time_of_last_spike_to_reach_synapse,
							float current_time_in_seconds,
							size_t total_number_of_synapses){

	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	while (idx < total_number_of_synapses) {
		// Decrememnt Spikes
		d_spikes_travelling_to_synapse[idx] -= 1;
		if (d_spikes_travelling_to_synapse[idx] == 0) {
			d_time_of_last_spike_to_reach_synapse[idx] = current_time_in_seconds;
		}
		idx += blockDim.x * gridDim.x;
	}
	__syncthreads();
}

void SpikingSynapses::update_synaptic_conductances(float timestep, float current_time_in_seconds) {

}

__global__ void move_spikes_towards_synapses_kernal(int* d_presynaptic_neuron_indices,
								int* d_delays,
								int* d_spikes_travelling_to_synapse,
								float* d_last_spike_time_of_each_neuron,
								float* d_input_neurons_last_spike_time,
								int* d_spikes_travelling_to_synapse_buffer,
								float current_time_in_seconds,
								size_t total_number_of_synapses){
	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	while (idx < total_number_of_synapses) {

		// Reduce the spikebuffer by 1
		d_spikes_travelling_to_synapse_buffer[idx] -= 1;

		// Get presynaptic neurons last spike time
		int presynaptic_neuron_index = d_presynaptic_neuron_indices[idx];
		bool presynaptic_is_input = PRESYNAPTIC_IS_INPUT(presynaptic_neuron_index);
		float presynaptic_neurons_last_spike_time = presynaptic_is_input ? d_input_neurons_last_spike_time[CORRECTED_PRESYNAPTIC_ID(presynaptic_neuron_index, presynaptic_is_input)] : d_last_spike_time_of_each_neuron[presynaptic_neuron_index];

		// If the presynaptic neuron has JUST fired, add spike to spikes_travelling_to_synapse or buffer, with delay time
		if (presynaptic_neurons_last_spike_time == current_time_in_seconds){

			if (d_spikes_travelling_to_synapse[idx] <= 0){
				d_spikes_travelling_to_synapse[idx] = d_delays[idx];
			} else if (d_spikes_travelling_to_synapse_buffer[idx] <= 0){
				d_spikes_travelling_to_synapse_buffer[idx] = d_delays[idx];
			}

		}

		// If there is spike in main array has expired, add buffer spike (if exists) to main array. Or set both to negative if no spikes.
		if (d_spikes_travelling_to_synapse[idx] <= 0) {
			if (d_spikes_travelling_to_synapse_buffer[idx] > 0) {
				d_spikes_travelling_to_synapse[idx] = d_spikes_travelling_to_synapse_buffer[idx];
			} else {
				d_spikes_travelling_to_synapse[idx] = -1;
				d_spikes_travelling_to_synapse_buffer[idx] = -1;
			}

		}

		// If the buffer has a smaller time than the spike, switch them
		if ((d_spikes_travelling_to_synapse_buffer[idx] > 0) && (d_spikes_travelling_to_synapse_buffer[idx] < d_spikes_travelling_to_synapse[idx])){
			int temp = d_spikes_travelling_to_synapse[idx];
			d_spikes_travelling_to_synapse[idx] = d_spikes_travelling_to_synapse_buffer[idx];
			d_spikes_travelling_to_synapse_buffer[idx] = temp;

		}

		idx += blockDim.x * gridDim.x;
	}
}