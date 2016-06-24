#include "SpikingNeurons.h"
#include <stdlib.h>
#include "../Helpers/CUDAErrorCheckHelpers.h"


// SpikingNeurons Constructor
SpikingNeurons::SpikingNeurons() {

	after_spike_reset_membrane_potentials_c = NULL;
	thresholds_for_action_potential_spikes = NULL;

	d_last_spike_time_of_each_neuron = NULL;
	d_membrane_potentials_v = NULL;
	d_thresholds_for_action_potential_spikes = NULL;
	d_resting_potentials = NULL;

}


// SpikingNeurons Destructor
SpikingNeurons::~SpikingNeurons() {
}


int SpikingNeurons::AddGroup(neuron_parameters_struct * group_params){
	
	int new_group_id = Neurons::AddGroup(group_params);

	spiking_neuron_parameters_struct * spiking_group_params = (spiking_neuron_parameters_struct*)group_params;

	after_spike_reset_membrane_potentials_c = (float*)realloc(after_spike_reset_membrane_potentials_c, (total_number_of_neurons*sizeof(float)));
	thresholds_for_action_potential_spikes = (float*)realloc(thresholds_for_action_potential_spikes, (total_number_of_neurons*sizeof(float)));

	for (int i = total_number_of_neurons - number_of_neurons_in_new_group; i < total_number_of_neurons; i++) {
		after_spike_reset_membrane_potentials_c[i] = spiking_group_params->resting_potential_v0;
		thresholds_for_action_potential_spikes[i] = spiking_group_params->threshold_for_action_potential_spike;
	}

	return new_group_id;
}


void SpikingNeurons::allocate_device_pointers() {

	Neurons::allocate_device_pointers();

	CudaSafeCall(cudaMalloc((void **)&d_last_spike_time_of_each_neuron, sizeof(float)*total_number_of_neurons));

	CudaSafeCall(cudaMalloc((void **)&d_membrane_potentials_v, sizeof(float)*total_number_of_neurons));
	CudaSafeCall(cudaMalloc((void **)&d_thresholds_for_action_potential_spikes, sizeof(float)*total_number_of_neurons));
	CudaSafeCall(cudaMalloc((void **)&d_resting_potentials, sizeof(float)*total_number_of_neurons));

}

void SpikingNeurons::reset_neurons() {

	Neurons::reset_neurons();

	CudaSafeCall(cudaMemset(d_last_spike_time_of_each_neuron, -1000.0f, total_number_of_neurons*sizeof(float)));

	CudaSafeCall(cudaMemcpy(d_membrane_potentials_v, after_spike_reset_membrane_potentials_c, sizeof(float)*total_number_of_neurons, cudaMemcpyHostToDevice));
	CudaSafeCall(cudaMemcpy(d_thresholds_for_action_potential_spikes, thresholds_for_action_potential_spikes, sizeof(float)*total_number_of_neurons, cudaMemcpyHostToDevice));
	CudaSafeCall(cudaMemcpy(d_resting_potentials, after_spike_reset_membrane_potentials_c, sizeof(float)*total_number_of_neurons, cudaMemcpyHostToDevice));

}


void SpikingNeurons::update_membrane_potentials(float timestep) {
	
}

void SpikingNeurons::check_for_neuron_spikes(float current_time_in_seconds, float timestep) {

	check_for_neuron_spikes_kernel<<<number_of_neuron_blocks_per_grid, threads_per_block>>>(d_membrane_potentials_v,
																	d_thresholds_for_action_potential_spikes,
																	d_resting_potentials,
																	d_last_spike_time_of_each_neuron,
																	current_time_in_seconds,
																	total_number_of_neurons);

	CudaCheckError();
}


// Spiking Neurons
__global__ void check_for_neuron_spikes_kernel(float *d_membrane_potentials_v,
								float *d_thresholds_for_action_potential_spikes,
								float *d_resting_potentials,
								float* d_last_spike_time_of_each_neuron,
								float current_time_in_seconds,
								size_t total_number_of_neurons) {

	// Get thread IDs
	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	while (idx < total_number_of_neurons) {

		if (d_membrane_potentials_v[idx] >= d_thresholds_for_action_potential_spikes[idx]) {

			// Set current time as last spike time of neuron
			d_last_spike_time_of_each_neuron[idx] = current_time_in_seconds;

			// Reset membrane potential
			d_membrane_potentials_v[idx] = d_resting_potentials[idx];

		}

		idx += blockDim.x * gridDim.x;
	}
	__syncthreads();

}
