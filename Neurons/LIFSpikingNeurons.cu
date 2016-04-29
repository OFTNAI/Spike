#include "LIFSpikingNeurons.h"
#include <stdlib.h>
#include <stdio.h>
#include "../Helpers/CUDAErrorCheckHelpers.h"


// LIFSpikingNeurons Constructor
LIFSpikingNeurons::LIFSpikingNeurons() {
	// param_a = NULL;
	// param_b = NULL;

	// d_param_a = NULL;
	// d_param_b = NULL;
}


// LIFSpikingNeurons Destructor
LIFSpikingNeurons::~LIFSpikingNeurons() {

}


int LIFSpikingNeurons::AddGroup(neuron_parameters_struct * group_params, int group_shape[2]){

	int new_group_id = SpikingNeurons::AddGroup(group_params, group_shape);

	lif_spiking_neuron_parameters_struct * lif_spiking_group_params = (lif_spiking_neuron_parameters_struct*)group_params;

	// param_a = (float*)realloc(param_a, (total_number_of_neurons*sizeof(float)));
	// param_b = (float*)realloc(param_b, (total_number_of_neurons*sizeof(float)));

	for (int i = total_number_of_neurons - number_of_neurons_in_new_group; i < total_number_of_neurons; i++) {
		// param_a[i] = izhikevich_spiking_group_params->parama;
		// param_b[i] = izhikevich_spiking_group_params->paramb;
	}

	return new_group_id;
}


void LIFSpikingNeurons::initialise_device_pointers() {
 	
 	SpikingNeurons::initialise_device_pointers();

 	// CudaSafeCall(cudaMalloc((void **)&d_param_a, sizeof(float)*total_number_of_neurons));
 	// CudaSafeCall(cudaMalloc((void **)&d_param_b, sizeof(float)*total_number_of_neurons));
 	
}

void LIFSpikingNeurons::reset_neurons() {

	SpikingNeurons::reset_neurons();	

	// CudaSafeCall(cudaMemcpy(d_param_a, param_a, sizeof(float)*total_number_of_neurons, cudaMemcpyHostToDevice));
	// CudaSafeCall(cudaMemcpy(d_param_b, param_b, sizeof(float)*total_number_of_neurons, cudaMemcpyHostToDevice));
}


__global__ void lif_update_membrane_potentials(float *d_membrane_potentials_v,
								float* d_current_injections,
								float timestep,
								size_t total_number_of_neurons);


void LIFSpikingNeurons::update_membrane_potentials(float timestep) {

	lif_update_membrane_potentials<<<number_of_neuron_blocks_per_grid, threads_per_block>>>(d_membrane_potentials_v,
																	d_current_injections,
																	timestep,
																	total_number_of_neurons);

	CudaCheckError();
}


// State Update
__global__ void lif_update_membrane_potentials(float *d_membrane_potentials_v,
								float* d_current_injections,
								float timestep,
								size_t total_number_of_neurons){

	
	// // Get thread IDs
	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	if (idx < total_number_of_neurons) {

		// We require the equation timestep in ms:
		float eqtimestep = timestep*1000.0f;

		float membrane_potential_Vi = d_membrane_potentials_v[idx];
		float current_injection_Ii = d_current_injections[idx];
		float temp_resting_potential_V0 = -74.0; // Same as after_spike_reset_membrane_potential ???
		float temp_membrane_resistance_R = 40000000.0f;
		float new_membrane_potential = eqtimestep * (temp_resting_potential_V0 - temp_membrane_resistance_R * current_injection_Ii) + (1 - eqtimestep) * membrane_potential_Vi;

		// if (idx == 10) {
		// 	printf("%f\n", new_membrane_potential);
		// }

		d_membrane_potentials_v[idx] = new_membrane_potential;

	}
	__syncthreads();
}


