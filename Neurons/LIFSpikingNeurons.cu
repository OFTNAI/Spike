#include "LIFSpikingNeurons.h"
#include <stdlib.h>
#include <stdio.h>
#include "../Helpers/CUDAErrorCheckHelpers.h"


// LIFSpikingNeurons Constructor
LIFSpikingNeurons::LIFSpikingNeurons() {
	
	

}


// LIFSpikingNeurons Destructor
LIFSpikingNeurons::~LIFSpikingNeurons() {
	
}


int LIFSpikingNeurons::AddGroup(neuron_parameters_struct * group_params, int group_shape[2]){

	int new_group_id = SpikingNeurons::AddGroup(group_params, group_shape);

	lif_spiking_neuron_parameters_struct * lif_spiking_group_params = (lif_spiking_neuron_parameters_struct*)group_params;

	// for (int i = total_number_of_neurons - number_of_neurons_in_new_group; i < total_number_of_neurons; i++) {
	// }

	return new_group_id;
}


void LIFSpikingNeurons::allocate_device_pointers() {
 	
 	SpikingNeurons::allocate_device_pointers();


}

void LIFSpikingNeurons::reset_neurons() {

	SpikingNeurons::reset_neurons();	
}


__global__ void lif_update_membrane_potentials(float *d_membrane_potentials_v,
								float* d_current_injections,
								float timestep,
								size_t total_number_of_neurons);

__global__ void lif_update_postsynaptic_activities_kernal(float timestep,
								size_t total_number_of_neurons,
								float * d_recent_postsynaptic_activities_D,
								float * d_last_spike_time_of_each_neuron,
								float current_time_in_seconds);


void LIFSpikingNeurons::update_membrane_potentials(float timestep) {

	lif_update_membrane_potentials<<<number_of_neuron_blocks_per_grid, threads_per_block>>>(d_membrane_potentials_v,
																	d_current_injections,
																	timestep,
																	total_number_of_neurons);

	CudaCheckError();
}

void LIFSpikingNeurons::update_postsynaptic_activities(float timestep, float current_time_in_seconds) {

	lif_update_postsynaptic_activities_kernal<<<number_of_neuron_blocks_per_grid, threads_per_block>>>(timestep,
								total_number_of_neurons,
								d_recent_postsynaptic_activities_D,
								d_last_spike_time_of_each_neuron,
								current_time_in_seconds);

}


// State Update
__global__ void lif_update_membrane_potentials(float *d_membrane_potentials_v,
								float* d_current_injections,
								float timestep,
								size_t total_number_of_neurons){

	
	// // Get thread IDs
	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	if (idx < total_number_of_neurons) {

		float membrane_time_constant_in_seconds = 0.02;
		float equation_constant = timestep / membrane_time_constant_in_seconds;

		float membrane_potential_Vi = d_membrane_potentials_v[idx];
		float current_injection_Ii = d_current_injections[idx];
		float temp_resting_potential_V0 = -74.0; // Same as after_spike_reset_membrane_potential ???
		float temp_membrane_resistance_R = 40000000.0f;

		float new_membrane_potential = equation_constant * (temp_resting_potential_V0 + temp_membrane_resistance_R * current_injection_Ii) + (1 - equation_constant) * membrane_potential_Vi;

		d_membrane_potentials_v[idx] = new_membrane_potential;

	}
	__syncthreads();
}

__global__ void lif_update_postsynaptic_activities_kernal(float timestep,
								size_t total_number_of_neurons,
								float * d_recent_postsynaptic_activities_D,
								float * d_last_spike_time_of_each_neuron,
								float current_time_in_seconds) {

	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	if (idx < total_number_of_neurons) {

		// if (d_stdp[idx] == 1) {

			float recent_postsynaptic_activity_D = d_recent_postsynaptic_activities_D[idx];
			float decay_term_tau_D = 0.03; // Should be variable between 0.005 and 0.125

			float new_recent_postsynaptic_activity_D = (1 - (timestep/decay_term_tau_D)) * recent_postsynaptic_activity_D;

			if (d_last_spike_time_of_each_neuron[idx] == current_time_in_seconds) {
				float model_parameter_alpha_D = 0.5;
				new_recent_postsynaptic_activity_D += timestep * model_parameter_alpha_D * (1 - recent_postsynaptic_activity_D);
			}

			d_recent_postsynaptic_activities_D[idx] = new_recent_postsynaptic_activity_D;

		// }

	}
}


