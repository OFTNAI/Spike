#include "SpikingNeurons.h"
#include <stdlib.h>
#include "CUDAErrorCheckHelpers.h"


// SpikingNeurons Constructor
SpikingNeurons::SpikingNeurons() {
	states_v = NULL;
	states_u = NULL;
	param_c = NULL;
	param_d = NULL;

	// d_last_spike_time = NULL;
	d_states_v = NULL;
	d_states_u = NULL;
	d_param_c = NULL;
	d_param_d = NULL;
}


// SpikingNeurons Destructor
SpikingNeurons::~SpikingNeurons() {

}


int SpikingNeurons::AddGroup(neuron_parameters_struct * group_params, int group_shape[2]){
	
	int new_group_id = Neurons::AddGroup(group_params, group_shape);

	spiking_neuron_parameters_struct * spiking_group_params = (spiking_neuron_parameters_struct*)group_params;

	states_v = (float*)realloc(states_v, (total_number_of_neurons*sizeof(float)));
	states_u = (float*)realloc(states_u, (total_number_of_neurons*sizeof(float)));
	param_c = (float*)realloc(param_c, (total_number_of_neurons*sizeof(float)));
	param_d = (float*)realloc(param_d, (total_number_of_neurons*sizeof(float)));

	for (int i = total_number_of_neurons - number_of_neurons_in_new_group; i < total_number_of_neurons; i++) {
		states_v[i] = spiking_group_params->state_v;
		states_u[i] = spiking_group_params->state_u;
		param_c[i] = spiking_group_params->paramc;
		param_d[i] = spiking_group_params->paramd;
	}

	return new_group_id;
}


void SpikingNeurons::initialise_device_pointers() {

	Neurons::initialise_device_pointers();

	CudaSafeCall(cudaMalloc((void **)&d_states_v, sizeof(float)*total_number_of_neurons));
 	CudaSafeCall(cudaMalloc((void **)&d_states_u, sizeof(float)*total_number_of_neurons));
 	CudaSafeCall(cudaMalloc((void **)&d_param_c, sizeof(float)*total_number_of_neurons));
 	CudaSafeCall(cudaMalloc((void **)&d_param_d, sizeof(float)*total_number_of_neurons));


	SpikingNeurons::reset_neuron_variables_and_spikes();
}

void SpikingNeurons::reset_neuron_variables_and_spikes() {

	CudaSafeCall(cudaMemcpy(d_states_v, states_v, sizeof(float)*total_number_of_neurons, cudaMemcpyHostToDevice));
	CudaSafeCall(cudaMemcpy(d_states_u, states_u, sizeof(float)*total_number_of_neurons, cudaMemcpyHostToDevice));
	CudaSafeCall(cudaMemcpy(d_param_c, param_c, sizeof(float)*total_number_of_neurons, cudaMemcpyHostToDevice));
	CudaSafeCall(cudaMemcpy(d_param_d, param_d, sizeof(float)*total_number_of_neurons, cudaMemcpyHostToDevice));
	
}


void SpikingNeurons::set_threads_per_block_and_blocks_per_grid(int threads) {
	
	Neurons::set_threads_per_block_and_blocks_per_grid(threads);

}



void SpikingNeurons::state_update_wrapper(float timestep) {
	
}



__global__ void check_for_neuron_spikes(float *d_states_v,
								float *d_states_u,
								float *d_param_c,
								float *d_param_d,
								float* d_last_spike_time,
								float currtime,
								size_t total_number_of_neurons);


void SpikingNeurons::check_for_neuron_spikes_wrapper(float currtime) {

	check_for_neuron_spikes<<<number_of_neuron_blocks_per_grid, threads_per_block>>>(d_states_v,
																	d_states_u,
																	d_param_c,
																	d_param_d,
																	d_last_spike_time,
																	currtime,
																	total_number_of_neurons);

	CudaCheckError();
}


// Spiking Neurons
__global__ void check_for_neuron_spikes(float *d_states_v,
								float *d_states_u,
								float *d_param_c,
								float *d_param_d,
								float* d_last_spike_time,
								float currtime,
								size_t total_number_of_neurons) {

	// Get thread IDs
	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	if (idx < total_number_of_neurons) {
		// First checking if neuron has spiked:
		if (d_states_v[idx] >= 30.0f){
			// Reset the values of these neurons
			d_states_v[idx] = d_param_c[idx];
			d_states_u[idx] += d_param_d[idx];
			// Update the last spike times of these neurons
			d_last_spike_time[idx] = currtime;
		}
	}
	__syncthreads();

}
