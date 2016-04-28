#include "SpikingNeurons.h"
#include <stdlib.h>
#include "../Helpers/CUDAErrorCheckHelpers.h"


// SpikingNeurons Constructor
SpikingNeurons::SpikingNeurons() {

	after_spike_reset_membrane_potentials_c = NULL;
	param_d = NULL;

	d_last_spike_times = NULL;
	d_membrane_potentials_v = NULL;
	d_states_u = NULL;
	d_after_spike_reset_membrane_potentials_c = NULL;
	d_param_d = NULL;
}


// SpikingNeurons Destructor
SpikingNeurons::~SpikingNeurons() {

}


int SpikingNeurons::AddGroup(neuron_parameters_struct * group_params, int group_shape[2]){
	
	int new_group_id = Neurons::AddGroup(group_params, group_shape);

	spiking_neuron_parameters_struct * spiking_group_params = (spiking_neuron_parameters_struct*)group_params;

	// states_v = (float*)realloc(states_v, (total_number_of_neurons*sizeof(float)));
	// states_u = (float*)realloc(states_u, (total_number_of_neurons*sizeof(float)));
	after_spike_reset_membrane_potentials_c = (float*)realloc(after_spike_reset_membrane_potentials_c, (total_number_of_neurons*sizeof(float)));
	param_d = (float*)realloc(param_d, (total_number_of_neurons*sizeof(float)));

	for (int i = total_number_of_neurons - number_of_neurons_in_new_group; i < total_number_of_neurons; i++) {
		// states_v[i] = spiking_group_params->state_v;
		// states_u[i] = spiking_group_params->state_u;
		after_spike_reset_membrane_potentials_c[i] = spiking_group_params->after_spike_reset_membrane_potential_c;
		param_d[i] = spiking_group_params->paramd;
	}

	return new_group_id;
}


void SpikingNeurons::initialise_device_pointers() {

	Neurons::initialise_device_pointers();

	CudaSafeCall(cudaMalloc((void **)&d_last_spike_times, sizeof(float)*total_number_of_neurons));

	CudaSafeCall(cudaMalloc((void **)&d_membrane_potentials_v, sizeof(float)*total_number_of_neurons));
 	CudaSafeCall(cudaMalloc((void **)&d_states_u, sizeof(float)*total_number_of_neurons));
 	CudaSafeCall(cudaMalloc((void **)&d_after_spike_reset_membrane_potentials_c, sizeof(float)*total_number_of_neurons));
 	CudaSafeCall(cudaMalloc((void **)&d_param_d, sizeof(float)*total_number_of_neurons));
}

void SpikingNeurons::reset_neurons() {

	Neurons::reset_neurons();

	CudaSafeCall(cudaMemset(d_last_spike_times, -1000.0f, total_number_of_neurons*sizeof(float)));

	CudaSafeCall(cudaMemcpy(d_membrane_potentials_v, after_spike_reset_membrane_potentials_c, sizeof(float)*total_number_of_neurons, cudaMemcpyHostToDevice));
	CudaSafeCall(cudaMemset(d_states_u, 0.0f, sizeof(float)*total_number_of_neurons));
	CudaSafeCall(cudaMemcpy(d_after_spike_reset_membrane_potentials_c, after_spike_reset_membrane_potentials_c, sizeof(float)*total_number_of_neurons, cudaMemcpyHostToDevice));
	CudaSafeCall(cudaMemcpy(d_param_d, param_d, sizeof(float)*total_number_of_neurons, cudaMemcpyHostToDevice));
}


void SpikingNeurons::set_threads_per_block_and_blocks_per_grid(int threads) {
	
	Neurons::set_threads_per_block_and_blocks_per_grid(threads);

}



void SpikingNeurons::update_neuron_states(float timestep) {
	
}



__global__ void check_for_neuron_spikes_kernal(float *d_membrane_potentials_v,
								float *d_states_u,
								float *d_after_spike_reset_membrane_potentials_c,
								float *d_param_d,
								float* d_last_spike_times,
								float currtime,
								size_t total_number_of_neurons);


void SpikingNeurons::check_for_neuron_spikes(float currtime) {

	check_for_neuron_spikes_kernal<<<number_of_neuron_blocks_per_grid, threads_per_block>>>(d_membrane_potentials_v,
																	d_states_u,
																	d_after_spike_reset_membrane_potentials_c,
																	d_param_d,
																	d_last_spike_times,
																	currtime,
																	total_number_of_neurons);

	CudaCheckError();
}


// Spiking Neurons
__global__ void check_for_neuron_spikes_kernal(float *d_membrane_potentials_v,
								float *d_states_u,
								float *d_after_spike_reset_membrane_potentials_c,
								float *d_param_d,
								float* d_last_spike_times,
								float currtime,
								size_t total_number_of_neurons) {

	// Get thread IDs
	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	if (idx < total_number_of_neurons) {
		// First checking if neuron has spiked:
		// if (total_number_of_neurons == 20) {
		// 	printf("d_membrane_potentials_v[idx] = %f\n", d_membrane_potentials_v[idx]);
		// }
		if (d_membrane_potentials_v[idx] >= 30.0f){
			// Reset the values of these neurons
			d_membrane_potentials_v[idx] = d_after_spike_reset_membrane_potentials_c[idx];
			d_states_u[idx] += d_param_d[idx];
			// Update the last spike times of these neurons
			d_last_spike_times[idx] = currtime;
		}
	}
	__syncthreads();

}
