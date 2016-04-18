#include "PoissonSpikingNeurons.h"
#include <stdlib.h>
#include <stdio.h>
#include "CUDAErrorCheckHelpers.h"
#include <algorithm> // For random shuffle
using namespace std;


// PoissonSpikingNeurons Constructor
PoissonSpikingNeurons::PoissonSpikingNeurons() {

}


// PoissonSpikingNeurons Destructor
PoissonSpikingNeurons::~PoissonSpikingNeurons() {

}


int PoissonSpikingNeurons::AddGroupNew(neuron_parameters_struct * group_params, int group_shape[2]){
	
	int new_group_id = SpikingNeurons::AddGroupNew(group_params, group_shape);

	poisson_spiking_neuron_parameters_struct * poisson_spiking_group_params = (poisson_spiking_neuron_parameters_struct*)group_params;

	rates = (float*)realloc(rates, (total_number_of_neurons*sizeof(float)));

	for (int i = 0; i < total_number_of_neurons; i++) {
		rates[i] = poisson_spiking_group_params->rate;
	}

	return -1 * new_group_id;

}

void PoissonSpikingNeurons::initialise_device_pointersNew() {

	SpikingNeurons::initialise_device_pointersNew();

	CudaSafeCall(cudaMalloc((void **)&d_rates, sizeof(float)*total_number_of_neurons));
	CudaSafeCall(cudaMalloc((void**) &d_states, total_number_of_neurons*sizeof(curandState_t)));

	PoissonSpikingNeurons::reset_input_variables();
}


void PoissonSpikingNeurons::reset_input_variables() {
	CudaSafeCall(cudaMemcpy(d_rates, rates, sizeof(float)*total_number_of_neurons, cudaMemcpyHostToDevice));
	CudaSafeCall(cudaMemcpy(d_states_v, states_v, sizeof(float)*total_number_of_neurons, cudaMemcpyHostToDevice));
}




__global__ void generate_random_states(unsigned int seed, curandState_t* d_states, size_t total_number_of_neurons);


void PoissonSpikingNeurons::generate_random_states_wrapper() {
	
	generate_random_states<<<number_of_neuron_blocks_per_grid, threads_per_block>>>(42, d_states, total_number_of_neurons);

	CudaCheckError();
}


__global__ void generate_random_states(unsigned int seed, curandState_t* d_states, size_t total_number_of_neurons) {
	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	if (idx < total_number_of_neurons) {
		curand_init(seed, /* the seed can be the same for each core, here we pass the time in from the CPU */
					idx, /* the sequence number should be different for each core (unless you want all
							cores to get the same sequence of numbers for some reason - use thread id! */
 					0, /* the offset is how much extra we advance in the sequence for each call, can be 0 */
					&d_states[idx]);
	}
}





__global__ void update_poisson_state(curandState_t* d_states,
							float *d_rates,
							float *d_states_v,
							float *d_states_u,
							float timestep,
							size_t total_number_of_inputs);



void PoissonSpikingNeurons::update_poisson_state_wrapper(float timestep) {

	update_poisson_state<<<number_of_neuron_blocks_per_grid, threads_per_block>>>(d_states,
														d_rates,
														d_states_v,
														d_states_u,
														timestep,
														total_number_of_neurons);
	CudaCheckError();
}


__global__ void update_poisson_state(curandState_t* d_states,
							float *d_rates,
							float *d_states_v,
							float *d_states_u,
							float timestep,
							size_t total_number_of_inputs){

	int idx = threadIdx.x + blockIdx.x * blockDim.x;

	if (idx < total_number_of_inputs){

		float random_float = curand_uniform(&d_states[idx]);;
		
		// if the randomnumber is LT the rate
		if (random_float < (d_rates[idx]*timestep)){
			d_states_v[idx] = 35.0f;
			d_states_u[idx] = 0.0f;
		} else if (d_rates[idx] != 0.0f) {
			d_states_v[idx] = -70.0f;
			d_states_u[idx] = 0.0f;
		}

	}
	__syncthreads();
}

