#include "IzhikevichSpikingNeurons.h"
#include <stdlib.h>
#include <stdio.h>
#include "../Helpers/CUDAErrorCheckHelpers.h"


// IzhikevichSpikingNeurons Constructor
IzhikevichSpikingNeurons::IzhikevichSpikingNeurons() {
	
	param_a = NULL;
	param_b = NULL;
	param_d = NULL;

	d_param_a = NULL;
	d_param_b = NULL;
	d_param_d = NULL;

	d_states_u = NULL;
	
}


// IzhikevichSpikingNeurons Destructor
IzhikevichSpikingNeurons::~IzhikevichSpikingNeurons() {

}


int IzhikevichSpikingNeurons::AddGroup(neuron_parameters_struct * group_params){

	int new_group_id = SpikingNeurons::AddGroup(group_params);

	izhikevich_spiking_neuron_parameters_struct * izhikevich_spiking_group_params = (izhikevich_spiking_neuron_parameters_struct*)group_params;

	param_a = (float*)realloc(param_a, (total_number_of_neurons*sizeof(float)));
	param_b = (float*)realloc(param_b, (total_number_of_neurons*sizeof(float)));
	param_d = (float*)realloc(param_d, (total_number_of_neurons*sizeof(float)));


	for (int i = total_number_of_neurons - number_of_neurons_in_new_group; i < total_number_of_neurons; i++) {
		param_a[i] = izhikevich_spiking_group_params->parama;
		param_b[i] = izhikevich_spiking_group_params->paramb;
		param_d[i] = izhikevich_spiking_group_params->paramd;
	}

	return new_group_id;
}


void IzhikevichSpikingNeurons::allocate_device_pointers() {
 	
 	SpikingNeurons::allocate_device_pointers();

 	CudaSafeCall(cudaMalloc((void **)&d_param_a, sizeof(float)*total_number_of_neurons));
 	CudaSafeCall(cudaMalloc((void **)&d_param_b, sizeof(float)*total_number_of_neurons));
 	CudaSafeCall(cudaMalloc((void **)&d_param_d, sizeof(float)*total_number_of_neurons));

 	CudaSafeCall(cudaMalloc((void **)&d_states_u, sizeof(float)*total_number_of_neurons));
 	
 	
}

void IzhikevichSpikingNeurons::reset_neurons() {

	SpikingNeurons::reset_neurons();	

	CudaSafeCall(cudaMemcpy(d_param_a, param_a, sizeof(float)*total_number_of_neurons, cudaMemcpyHostToDevice));
	CudaSafeCall(cudaMemcpy(d_param_b, param_b, sizeof(float)*total_number_of_neurons, cudaMemcpyHostToDevice));
	CudaSafeCall(cudaMemcpy(d_param_d, param_d, sizeof(float)*total_number_of_neurons, cudaMemcpyHostToDevice));
	
	CudaSafeCall(cudaMemset(d_states_u, 0.0f, sizeof(float)*total_number_of_neurons));

}


// GPU Kernel Class Wrappers

void IzhikevichSpikingNeurons::check_for_neuron_spikes(float current_time_in_seconds, float timestep) {
	SpikingNeurons::check_for_neuron_spikes(current_time_in_seconds, timestep);

	reset_states_u_after_spikes_kernel<<<number_of_neuron_blocks_per_grid, threads_per_block>>>(d_states_u,
								d_param_d,
								d_last_spike_time_of_each_neuron,
								current_time_in_seconds,
								total_number_of_neurons);
}

void IzhikevichSpikingNeurons::update_membrane_potentials(float timestep) {

	izhikevich_update_membrane_potentials_kernel<<<number_of_neuron_blocks_per_grid, threads_per_block>>>(d_membrane_potentials_v,
																	d_states_u,
																	d_param_a,
																	d_param_b,
																	d_current_injections,
																	timestep,
																	total_number_of_neurons);

	CudaCheckError();
}


// GPU Kernels

__global__ void reset_states_u_after_spikes_kernel(float *d_states_u,
								float * d_param_d,
								float* d_last_spike_time_of_each_neuron,
								float current_time_in_seconds,
								size_t total_number_of_neurons) {
	
	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	while (idx < total_number_of_neurons) {
		if (d_last_spike_time_of_each_neuron[idx] == current_time_in_seconds) {

			d_states_u[idx] += d_param_d[idx];

		}
		idx += blockDim.x * gridDim.x;
	}
	__syncthreads();
}


__global__ void izhikevich_update_membrane_potentials_kernel(float *d_membrane_potentials_v,
								float *d_states_u,
								float *d_param_a,
								float *d_param_b,
								float* d_current_injections,
								float timestep,
								size_t total_number_of_neurons) {

	// We require the equation timestep in ms:
	float eqtimestep = timestep*1000.0f;
	// Get thread IDs
	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	if (idx < total_number_of_neurons) {
		// Update the neuron states according to the Izhikevich equations
		float v_update = 0.04f*d_membrane_potentials_v[idx]*d_membrane_potentials_v[idx] 
							+ 5.0f*d_membrane_potentials_v[idx]
							+ 140 
							- d_states_u[idx]
							+ d_current_injections[idx];

		d_membrane_potentials_v[idx] += eqtimestep*v_update;
		d_states_u[idx] += eqtimestep*(d_param_a[idx] * (d_param_b[idx] * d_membrane_potentials_v[idx] - 
							d_states_u[idx]));
	}
	__syncthreads();
}


