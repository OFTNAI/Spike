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


	// printf("POISSON  GROUP ID: %d\n", new_group_id);
	return -1 * new_group_id - 1;

}

void PoissonSpikingNeurons::initialise_device_pointers() {

	SpikingNeurons::initialise_device_pointers();

	CudaSafeCall(cudaMalloc((void **)&d_rates, sizeof(float)*total_number_of_neurons));
	CudaSafeCall(cudaMalloc((void**) &d_states, sizeof(curandState_t)*total_number_of_neurons));

}


void PoissonSpikingNeurons::reset_neurons() {

	SpikingNeurons::reset_neurons();

	CudaSafeCall(cudaMemcpy(d_rates, rates, sizeof(float)*total_number_of_neurons, cudaMemcpyHostToDevice));
	CudaSafeCall(cudaMemset(d_states, -1000.0f, sizeof(float)*total_number_of_neurons));
}



__global__ void generate_random_states_kernal(unsigned int seed, curandState_t* d_states, size_t total_number_of_neurons);


void PoissonSpikingNeurons::generate_random_states() {
	
	generate_random_states_kernal<<<number_of_neuron_blocks_per_grid, threads_per_block>>>(42, d_states, total_number_of_neurons);

	CudaCheckError();
}


__global__ void generate_random_states_kernal(unsigned int seed, curandState_t* d_states, size_t total_number_of_neurons) {
	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	if (idx < total_number_of_neurons) {
		curand_init(seed, /* the seed can be the same for each core, here we pass the time in from the CPU */
					idx, /* the sequence number should be different for each core (unless you want all
							cores to get the same sequence of numbers for some reason - use thread id! */
 					0, /* the offset is how much extra we advance in the sequence for each call, can be 0 */
					&d_states[idx]);
	}
}


__global__ void update_poisson_states_kernal(curandState_t* d_states,
							float *d_rates,
							float *d_membrane_potentials_v,
							float *d_states_u,
							float timestep,
							size_t total_number_of_inputs);


void PoissonSpikingNeurons::update_poisson_states(float timestep) {

	update_poisson_states_kernal<<<number_of_neuron_blocks_per_grid, threads_per_block>>>(d_states,
														d_rates,
														d_membrane_potentials_v,
														d_states_u,
														timestep,
														total_number_of_neurons);
	CudaCheckError();
}


__global__ void update_poisson_states_kernal(curandState_t* d_states,
							float *d_rates,
							float *d_membrane_potentials_v,
							float *d_states_u,
							float timestep,
							size_t total_number_of_inputs){

	int idx = threadIdx.x + blockIdx.x * blockDim.x;

	if (idx < total_number_of_inputs){

		float random_float = curand_uniform(&d_states[idx]);;

		// if the randomnumber is LT the rate
		if (random_float < (d_rates[idx]*timestep)){
			d_membrane_potentials_v[idx] = 35.0f;
			d_states_u[idx] = 0.0f;
		} else if (d_rates[idx] != 0.0f) {
			d_membrane_potentials_v[idx] = -70.0f;
			d_states_u[idx] = 0.0f;
		}

	}
	__syncthreads();
}

