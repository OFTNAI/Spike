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
	d_states = NULL;

	total_number_of_input_images = 1;

	random_state_manager = NULL;
}


// PoissonSpikingNeurons Destructor
PoissonSpikingNeurons::~PoissonSpikingNeurons() {

}


int PoissonSpikingNeurons::AddGroup(neuron_parameters_struct * group_params, int group_shape[2]){

	int new_group_id = SpikingNeurons::AddGroup(group_params, group_shape);

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
	CudaSafeCall(cudaMalloc((void**) &d_states, sizeof(curandState_t)*total_number_of_neurons));

}


void PoissonSpikingNeurons::reset_neurons() {

	SpikingNeurons::reset_neurons();

	CudaSafeCall(cudaMemcpy(d_rates, rates, sizeof(float)*total_number_of_neurons, cudaMemcpyHostToDevice));
	CudaSafeCall(cudaMemset(d_states, -1000.0f, sizeof(float)*total_number_of_neurons));
}


void PoissonSpikingNeurons::set_threads_per_block_and_blocks_per_grid(int threads) {
	
	SpikingNeurons::set_threads_per_block_and_blocks_per_grid(threads);

}


void PoissonSpikingNeurons::generate_random_states() {
	
	printf("Generating input neuron random states\n");

	if (random_state_manager == NULL) {
		random_state_manager = new RandomStateManager();
		random_state_manager->set_up_random_states(128, 64, 9);
	}
}


void PoissonSpikingNeurons::update_membrane_potentials(float timestep) {

	poisson_update_membrane_potentials_kernal<<<random_state_manager->block_dimensions, random_state_manager->threads_per_block>>>(random_state_manager->d_states,
														d_rates,
														d_membrane_potentials_v,
														timestep,
														d_thresholds_for_action_potential_spikes,
														total_number_of_neurons);

	CudaCheckError();
}


__global__ void poisson_update_membrane_potentials_kernal(curandState_t* d_states,
							float *d_rates,
							float *d_membrane_potentials_v,
							float timestep,
							float * d_thresholds_for_action_potential_spikes,
							size_t total_number_of_inputs){

	 
	int t_idx = threadIdx.x + blockIdx.x * blockDim.x;
	int idx = t_idx;
	while (idx < total_number_of_inputs){

		// Creates random float between 0 and 1 from uniform distribution
		// d_states effectively provides a different seed for each thread
		// curand_uniform produces different float every time you call it
		float random_float = curand_uniform(&d_states[t_idx]);

		// if the randomnumber is less than the rate
		if (random_float < (d_rates[idx] * timestep)){

			// Puts membrane potential above default spiking threshold
			d_membrane_potentials_v[idx] = d_thresholds_for_action_potential_spikes[idx] + 0.02;

		} 

		idx += blockDim.x * gridDim.x;

	}
	__syncthreads();
}

