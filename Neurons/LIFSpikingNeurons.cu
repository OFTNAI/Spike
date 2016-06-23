#include "LIFSpikingNeurons.h"
#include <stdlib.h>
#include <stdio.h>
#include "../Helpers/CUDAErrorCheckHelpers.h"


// LIFSpikingNeurons Constructor
LIFSpikingNeurons::LIFSpikingNeurons() {
	
	membrane_time_constants_tau_m = NULL;
	membrane_resistances_R = NULL;

	d_membrane_time_constants_tau_m = NULL;
	d_membrane_resistances_R = NULL;
}


// LIFSpikingNeurons Destructor
LIFSpikingNeurons::~LIFSpikingNeurons() {
	
}


int LIFSpikingNeurons::AddGroup(neuron_parameters_struct * group_params){

	int new_group_id = SpikingNeurons::AddGroup(group_params);

	lif_spiking_neuron_parameters_struct * lif_spiking_group_params = (lif_spiking_neuron_parameters_struct*)group_params;

	membrane_time_constants_tau_m = (float*)realloc(membrane_time_constants_tau_m, total_number_of_neurons*sizeof(float));
	membrane_resistances_R = (float*)realloc(membrane_resistances_R, total_number_of_neurons*sizeof(float));

	float membrane_time_constant_tau_m = lif_spiking_group_params->somatic_capcitance_Cm / lif_spiking_group_params->somatic_leakage_conductance_g0;
	float membrane_resistance_R = 1 / lif_spiking_group_params->somatic_leakage_conductance_g0;
	
	for (int i = total_number_of_neurons - number_of_neurons_in_new_group; i < total_number_of_neurons; i++) {
		membrane_time_constants_tau_m[i] = membrane_time_constant_tau_m;
		membrane_resistances_R[i] = membrane_resistance_R;
	}

	return new_group_id;
}


void LIFSpikingNeurons::allocate_device_pointers() {
 	
 	SpikingNeurons::allocate_device_pointers();

 	CudaSafeCall(cudaMalloc((void **)&d_membrane_time_constants_tau_m, sizeof(float)*total_number_of_neurons));
 	CudaSafeCall(cudaMalloc((void **)&d_membrane_resistances_R, sizeof(float)*total_number_of_neurons));

}

void LIFSpikingNeurons::reset_neurons() {

	SpikingNeurons::reset_neurons();	

	CudaSafeCall(cudaMemcpy(d_membrane_time_constants_tau_m, membrane_time_constants_tau_m, sizeof(float)*total_number_of_neurons, cudaMemcpyHostToDevice));
	CudaSafeCall(cudaMemcpy(d_membrane_resistances_R, membrane_resistances_R, sizeof(float)*total_number_of_neurons, cudaMemcpyHostToDevice));
}






void LIFSpikingNeurons::update_membrane_potentials(float timestep) {

	lif_update_membrane_potentials<<<number_of_neuron_blocks_per_grid, threads_per_block>>>(d_membrane_potentials_v,
																	d_membrane_resistances_R,
																	d_membrane_time_constants_tau_m,
																	d_resting_potentials,
																	d_current_injections,
																	timestep,
																	total_number_of_neurons);

	CudaCheckError();
}


__global__ void lif_update_membrane_potentials(float *d_membrane_potentials_v,
								float * d_membrane_resistances_R,
								float * d_membrane_time_constants_tau_m,
								float * d_resting_potentials,
								float* d_current_injections,
								float timestep,
								size_t total_number_of_neurons){

	
	// // Get thread IDs
	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	while (idx < total_number_of_neurons) {

		float equation_constant = timestep / d_membrane_time_constants_tau_m[idx];
		float membrane_potential_Vi = d_membrane_potentials_v[idx];
		float current_injection_Ii = d_current_injections[idx];
		float resting_potential_V0 = d_resting_potentials[idx];
		float temp_membrane_resistance_R = d_membrane_resistances_R[idx];

		float new_membrane_potential = equation_constant * (resting_potential_V0 + temp_membrane_resistance_R * current_injection_Ii) + (1 - equation_constant) * membrane_potential_Vi;

		d_membrane_potentials_v[idx] = new_membrane_potential;

		idx += blockDim.x * gridDim.x;

	}
	__syncthreads();
}


