#include "SpikingNeurons.h"
#include <stdlib.h>
#include "../Helpers/CUDAErrorCheckHelpers.h"


// SpikingNeurons Constructor
SpikingNeurons::SpikingNeurons() {

	after_spike_reset_membrane_potentials_c = NULL;
	thresholds_for_action_potential_spikes = NULL;
	param_d = NULL;

	d_last_spike_time_of_each_neuron = NULL;
	d_membrane_potentials_v = NULL;
	d_thresholds_for_action_potential_spikes = NULL;
	d_after_spike_reset_membrane_potentials_c = NULL;

	d_states_u = NULL;
	d_param_d = NULL;

	recent_postsynaptic_activities_D = NULL;
	d_recent_postsynaptic_activities_D = NULL;
}


// SpikingNeurons Destructor
SpikingNeurons::~SpikingNeurons() {
	free(recent_postsynaptic_activities_D);
	CudaSafeCall(cudaFree(d_recent_postsynaptic_activities_D));
}


int SpikingNeurons::AddGroup(neuron_parameters_struct * group_params, int group_shape[2]){
	
	int new_group_id = Neurons::AddGroup(group_params, group_shape);

	spiking_neuron_parameters_struct * spiking_group_params = (spiking_neuron_parameters_struct*)group_params;

	after_spike_reset_membrane_potentials_c = (float*)realloc(after_spike_reset_membrane_potentials_c, (total_number_of_neurons*sizeof(float)));
	thresholds_for_action_potential_spikes = (float*)realloc(thresholds_for_action_potential_spikes, (total_number_of_neurons*sizeof(float)));
	param_d = (float*)realloc(param_d, (total_number_of_neurons*sizeof(float)));
	recent_postsynaptic_activities_D = (float*)realloc(recent_postsynaptic_activities_D, (total_number_of_neurons*sizeof(float)));

	for (int i = total_number_of_neurons - number_of_neurons_in_new_group; i < total_number_of_neurons; i++) {
		after_spike_reset_membrane_potentials_c[i] = spiking_group_params->after_spike_reset_membrane_potential_c;
		thresholds_for_action_potential_spikes[i] = spiking_group_params->threshold_for_action_potential_spike;

		//Izhikevich extra
		param_d[i] = spiking_group_params->paramd;

		//LIF extra
		recent_postsynaptic_activities_D[i] = 0.0f;
	}

	return new_group_id;
}


void SpikingNeurons::allocate_device_pointers() {

	Neurons::allocate_device_pointers();

	CudaSafeCall(cudaMalloc((void **)&d_last_spike_time_of_each_neuron, sizeof(float)*total_number_of_neurons));

	CudaSafeCall(cudaMalloc((void **)&d_membrane_potentials_v, sizeof(float)*total_number_of_neurons));
	CudaSafeCall(cudaMalloc((void **)&d_thresholds_for_action_potential_spikes, sizeof(float)*total_number_of_neurons));
	CudaSafeCall(cudaMalloc((void **)&d_after_spike_reset_membrane_potentials_c, sizeof(float)*total_number_of_neurons));
 	
 	//Izhikevich extra
 	CudaSafeCall(cudaMalloc((void **)&d_states_u, sizeof(float)*total_number_of_neurons));
 	CudaSafeCall(cudaMalloc((void **)&d_param_d, sizeof(float)*total_number_of_neurons));

 	//LIF extra
 	 CudaSafeCall(cudaMalloc((void **)&d_recent_postsynaptic_activities_D, sizeof(float)*total_number_of_neurons));

}

void SpikingNeurons::reset_neurons() {

	Neurons::reset_neurons();

	CudaSafeCall(cudaMemset(d_last_spike_time_of_each_neuron, -1000.0f, total_number_of_neurons*sizeof(float)));

	CudaSafeCall(cudaMemcpy(d_membrane_potentials_v, after_spike_reset_membrane_potentials_c, sizeof(float)*total_number_of_neurons, cudaMemcpyHostToDevice));
	CudaSafeCall(cudaMemcpy(d_thresholds_for_action_potential_spikes, thresholds_for_action_potential_spikes, sizeof(float)*total_number_of_neurons, cudaMemcpyHostToDevice));
	CudaSafeCall(cudaMemcpy(d_after_spike_reset_membrane_potentials_c, after_spike_reset_membrane_potentials_c, sizeof(float)*total_number_of_neurons, cudaMemcpyHostToDevice));

	//Izhikevich extra
	CudaSafeCall(cudaMemset(d_states_u, 0.0f, sizeof(float)*total_number_of_neurons));
	CudaSafeCall(cudaMemcpy(d_param_d, param_d, sizeof(float)*total_number_of_neurons, cudaMemcpyHostToDevice));

	//LIF extra
	CudaSafeCall(cudaMemcpy(d_recent_postsynaptic_activities_D, recent_postsynaptic_activities_D, sizeof(float)*total_number_of_neurons, cudaMemcpyHostToDevice));

}


void SpikingNeurons::set_threads_per_block_and_blocks_per_grid(int threads) {
	
	Neurons::set_threads_per_block_and_blocks_per_grid(threads);

}



void SpikingNeurons::update_membrane_potentials(float timestep) {
	
}

void SpikingNeurons::update_postsynaptic_activities(float timestep, float current_time_in_seconds) {
	
}


__global__ void check_for_neuron_spikes_kernal(float *d_membrane_potentials_v,
								float *d_thresholds_for_action_potential_spikes,
								float *d_states_u,
								float *d_after_spike_reset_membrane_potentials_c,
								float *d_param_d,
								float* d_last_spike_time_of_each_neuron,
								float currtime,
								size_t total_number_of_neurons);


void SpikingNeurons::check_for_neuron_spikes(float currtime) {

	check_for_neuron_spikes_kernal<<<number_of_neuron_blocks_per_grid, threads_per_block>>>(d_membrane_potentials_v,
																	d_thresholds_for_action_potential_spikes,
																	d_states_u,
																	d_after_spike_reset_membrane_potentials_c,
																	d_param_d,
																	d_last_spike_time_of_each_neuron,
																	currtime,
																	total_number_of_neurons);

	CudaCheckError();
}


// Spiking Neurons
__global__ void check_for_neuron_spikes_kernal(float *d_membrane_potentials_v,
								float *d_thresholds_for_action_potential_spikes,
								float *d_states_u,
								float *d_after_spike_reset_membrane_potentials_c,
								float *d_param_d,
								float* d_last_spike_time_of_each_neuron,
								float currtime,
								size_t total_number_of_neurons) {

	// Get thread IDs
	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	while (idx < total_number_of_neurons) {

		// First checking if neuron has spiked:
		if (d_membrane_potentials_v[idx] >= d_thresholds_for_action_potential_spikes[idx]){

			// Set current time as last spike time of neuron
			d_last_spike_time_of_each_neuron[idx] = currtime;

			// Reset membrane potential
			d_membrane_potentials_v[idx] = d_after_spike_reset_membrane_potentials_c[idx];

			//Izhikevich extra reset
			d_states_u[idx] += d_param_d[idx];
			
		}

		idx += blockDim.x * gridDim.x;
	}
	__syncthreads();

}
