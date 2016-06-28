#include "GeneratorInputSpikingNeurons.h"
#include <stdlib.h>
#include <stdio.h>
#include "../Helpers/CUDAErrorCheckHelpers.h"


// GeneratorInputSpikingNeurons Constructor
GeneratorInputSpikingNeurons::GeneratorInputSpikingNeurons() {
	neuron_id_matrix_for_stimuli = NULL;
	spike_times_matrix_for_stimuli = NULL;
	number_of_spikes_in_stimuli = NULL;

	d_neuron_ids_for_stimulus = NULL;
	d_spike_times_for_stimulus = NULL;

	length_of_longest_stimulus = 0;
}


// GeneratorInputSpikingNeurons Destructor
GeneratorInputSpikingNeurons::~GeneratorInputSpikingNeurons() {
	free(neuron_id_matrix_for_stimuli);
	free(spike_times_matrix_for_stimuli);
	free(number_of_spikes_in_stimuli);
	CudaSafeCall(cudaFree(d_neuron_ids_for_stimulus));
	CudaSafeCall(cudaFree(d_spike_times_for_stimulus));
}

// Add Group of given size as usual - nothing special in constructor
int GeneratorInputSpikingNeurons::AddGroup(neuron_parameters_struct * group_params){
	
	int new_group_id = InputSpikingNeurons::AddGroup(group_params);
	return CORRECTED_PRESYNAPTIC_ID(new_group_id, true);

}

// Allocate device pointers for the longest stimulus so that they do not need to be replaced
void GeneratorInputSpikingNeurons::allocate_device_pointers(int maximum_axonal_delay_in_timesteps, bool high_fidelity_spike_storage) {

	InputSpikingNeurons::allocate_device_pointers(maximum_axonal_delay_in_timesteps, high_fidelity_spike_storage);

	CudaSafeCall(cudaMalloc((void **)&d_neuron_ids_for_stimulus, sizeof(int)*length_of_longest_stimulus));
	CudaSafeCall(cudaMalloc((void **)&d_spike_times_for_stimulus, sizeof(float)*length_of_longest_stimulus));
}


void GeneratorInputSpikingNeurons::reset_neurons() {
	CudaSafeCall(cudaMemcpy(d_neuron_ids_for_stimulus, neuron_id_matrix_for_stimuli[current_stimulus_index], sizeof(int)*number_of_spikes_in_stimuli[current_stimulus_index], cudaMemcpyHostToDevice));
	CudaSafeCall(cudaMemcpy(d_spike_times_for_stimulus, spike_times_matrix_for_stimuli[current_stimulus_index], sizeof(float)*number_of_spikes_in_stimuli[current_stimulus_index], cudaMemcpyHostToDevice));
}

void GeneratorInputSpikingNeurons::set_threads_per_block_and_blocks_per_grid(int threads) {
	
	InputSpikingNeurons::set_threads_per_block_and_blocks_per_grid(threads);

	int genblocknum = (length_of_longest_stimulus + threads) / threads;
	number_of_neuron_blocks_per_grid.x = genblocknum;
}

void GeneratorInputSpikingNeurons::check_for_neuron_spikes(float current_time_in_seconds, float timestep) {

	check_for_generator_spikes_kernel<<<number_of_neuron_blocks_per_grid, threads_per_block>>>(
		d_neuron_ids_for_stimulus,
		d_spike_times_for_stimulus,
		d_last_spike_time_of_each_neuron,
		d_bitarray_of_neuron_spikes,
		bitarray_length,
		bitarray_maximum_axonal_delay_in_timesteps,
		current_time_in_seconds,
		timestep,
		number_of_spikes_in_stimuli[current_stimulus_index],
		high_fidelity_spike_flag);


	CudaCheckError();
}

void GeneratorInputSpikingNeurons::update_membrane_potentials(float timestep){
}

void GeneratorInputSpikingNeurons::AddStimulus(int spikenumber, int* ids, float* spiketimes){

	++total_number_of_input_stimuli;
	// If the number of spikes in this stimulus is larger than any other ...
	if (spikenumber > length_of_longest_stimulus){
		length_of_longest_stimulus = spikenumber;
	}

	number_of_spikes_in_stimuli = (int*)realloc(number_of_spikes_in_stimuli, sizeof(int)*total_number_of_input_stimuli);
	neuron_id_matrix_for_stimuli = (int**)realloc(neuron_id_matrix_for_stimuli, sizeof(int*)*total_number_of_input_stimuli);
	spike_times_matrix_for_stimuli = (float**)realloc(spike_times_matrix_for_stimuli, sizeof(float*)*total_number_of_input_stimuli);
	
	// Initialize matrices
	neuron_id_matrix_for_stimuli[total_number_of_input_stimuli - 1] = NULL;
	spike_times_matrix_for_stimuli[total_number_of_input_stimuli - 1] = NULL;
	number_of_spikes_in_stimuli[total_number_of_input_stimuli - 1] = 0;
	
	neuron_id_matrix_for_stimuli[total_number_of_input_stimuli - 1] = (int*)realloc(
		neuron_id_matrix_for_stimuli[total_number_of_input_stimuli - 1], 
		sizeof(int)*(spikenumber));
	spike_times_matrix_for_stimuli[total_number_of_input_stimuli - 1] = (float*)realloc(
		spike_times_matrix_for_stimuli[total_number_of_input_stimuli - 1], 
		sizeof(float)*(spikenumber));

	
	// Assign the genid values according to how many neurons exist already
	for (int i = 0; i < spikenumber; i++){
		spike_times_matrix_for_stimuli[total_number_of_input_stimuli - 1][i] = ids[i];
		spike_times_matrix_for_stimuli[total_number_of_input_stimuli - 1][i] = spiketimes[i];
	}
	// Increment the number of entries the generator population
	number_of_spikes_in_stimuli[total_number_of_input_stimuli - 1] = spikenumber;
	
}

// Spiking Neurons
__global__ void check_for_generator_spikes_kernel(int *d_neuron_ids_for_stimulus,
								float *d_spike_times_for_stimulus,
								float* d_last_spike_time_of_each_neuron,
								unsigned char* d_bitarray_of_neuron_spikes,
								int bitarray_length,
								int bitarray_maximum_axonal_delay_in_timesteps,
								float current_time_in_seconds,
								float timestep,
								size_t number_of_spikes_in_stimulus,
								bool high_fidelity_spike_flag) {

	// Get thread IDs
	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	while (idx < number_of_spikes_in_stimulus) {
		if (fabs(current_time_in_seconds - d_spike_times_for_stimulus[idx]) < 0.5 * timestep) {
			__syncthreads();
			d_last_spike_time_of_each_neuron[d_neuron_ids_for_stimulus[idx]] = current_time_in_seconds;

			if (high_fidelity_spike_flag){
				// Get start of the given neuron's bits
				int neuron_id_spike_store_start = d_neuron_ids_for_stimulus[idx] * bitarray_length;
				// Get offset depending upon the current timestep
				int offset_index = (int)(round((float)(current_time_in_seconds / timestep))) % bitarray_maximum_axonal_delay_in_timesteps;
				int offset_byte = offset_index / 8;
				int offset_bit_pos = offset_index - (8 * offset_byte);
				// Get the specific position at which we should be putting the current value
				unsigned char byte = d_bitarray_of_neuron_spikes[neuron_id_spike_store_start + offset_byte];
				// Set the specific bit in the byte to on 
				byte |= (1 << offset_bit_pos);
				// Assign the byte
				d_bitarray_of_neuron_spikes[neuron_id_spike_store_start + offset_byte] = byte;
			}
		}

		idx += blockDim.x * gridDim.x;
	}
	__syncthreads();
}
