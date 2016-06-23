#include "PoissonSpikingNeurons.h"
#include <stdlib.h>
#include <stdio.h>
#include "../Helpers/CUDAErrorCheckHelpers.h"
#include <algorithm> // For random shuffle
using namespace std;


// PoissonSpikingNeurons Constructor
PoissonSpikingNeurons::PoissonSpikingNeurons() {
	rates = NULL;
	d_rates = NULL;

	total_number_of_input_images = 1;
	current_stimulus_index = 0;
}


// PoissonSpikingNeurons Destructor
PoissonSpikingNeurons::~PoissonSpikingNeurons() {

}


int PoissonSpikingNeurons::AddGroup(neuron_parameters_struct * group_params){

	int new_group_id = SpikingNeurons::AddGroup(group_params);

	poisson_spiking_neuron_parameters_struct * poisson_spiking_group_params = (poisson_spiking_neuron_parameters_struct*)group_params;

	rates = (float*)realloc(rates, sizeof(float)*total_number_of_neurons);
	for (int i = total_number_of_neurons - number_of_neurons_in_new_group; i < total_number_of_neurons; i++) {
		rates[i] = poisson_spiking_group_params->rate;
	}

	return CORRECTED_PRESYNAPTIC_ID(new_group_id, true);

}

void PoissonSpikingNeurons::allocate_device_pointers() {

	SpikingNeurons::allocate_device_pointers();

	CudaSafeCall(cudaMalloc((void **)&d_rates, sizeof(float)*total_number_of_neurons));

}


void PoissonSpikingNeurons::reset_neurons() {

	SpikingNeurons::reset_neurons();

	CudaSafeCall(cudaMemcpy(d_rates, rates, sizeof(float)*total_number_of_neurons, cudaMemcpyHostToDevice));
}


void PoissonSpikingNeurons::set_threads_per_block_and_blocks_per_grid(int threads) {
	
	SpikingNeurons::set_threads_per_block_and_blocks_per_grid(threads);

}


void PoissonSpikingNeurons::update_membrane_potentials(float timestep) {

	poisson_update_membrane_potentials_kernel<<<RandomStateManager::instance()->block_dimensions, RandomStateManager::instance()->threads_per_block>>>(RandomStateManager::instance()->d_states,
														d_rates,
														d_membrane_potentials_v,
														timestep,
														d_thresholds_for_action_potential_spikes,
														total_number_of_neurons,
														current_stimulus_index);

	CudaCheckError();
}


__global__ void poisson_update_membrane_potentials_kernel(curandState_t* d_states,
							float *d_rates,
							float *d_membrane_potentials_v,
							float timestep,
							float * d_thresholds_for_action_potential_spikes,
							size_t total_number_of_input_neurons,
							int current_stimulus_index) {

	 
	int t_idx = threadIdx.x + blockIdx.x * blockDim.x;
	int idx = t_idx;
	while (idx < total_number_of_input_neurons){

		int rate_index = (total_number_of_input_neurons * current_stimulus_index) + idx;

		float rate = d_rates[rate_index];

		if (rate > 0.1) {

			// Creates random float between 0 and 1 from uniform distribution
			// d_states effectively provides a different seed for each thread
			// curand_uniform produces different float every time you call it
			float random_float = curand_uniform(&d_states[t_idx]);
			
			// if the randomnumber is less than the rate
			if (random_float < (rate * timestep)) {

				// Puts membrane potential above default spiking threshold
				d_membrane_potentials_v[idx] = d_thresholds_for_action_potential_spikes[idx] + 0.02;

			} 

		}

		idx += blockDim.x * gridDim.x;

	}
	__syncthreads();
}

