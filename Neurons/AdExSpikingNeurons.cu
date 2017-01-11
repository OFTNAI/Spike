#include "AdExSpikingNeurons.h"
#include <stdlib.h>
#include <stdio.h>
#include "../Helpers/CUDAErrorCheckHelpers.h"


// AdExSpikingNeurons Constructor
AdExSpikingNeurons::AdExSpikingNeurons() {

	adaptation_values_w = NULL;
	membrane_capacitances_Cm = NULL;
	membrane_leakage_conductances_g0 = NULL;
	leak_reversal_potentials_E_L = NULL;
	slope_factors_Delta_T = NULL;
	adaptation_coupling_coefficients_a = NULL;
	adaptation_time_constants_tau_w = NULL;
	adaptation_changes_b = NULL;

	d_adaptation_values_w = NULL;
	d_membrane_capacitances_Cm = NULL;
	d_membrane_leakage_conductances_g0 = NULL;
	d_leak_reversal_potentials_E_L = NULL;
	d_slope_factors_Delta_T = NULL;
	d_adaptation_coupling_coefficients_a = NULL;
	d_adaptation_time_constants_tau_w = NULL;
	d_adaptation_changes_b = NULL;
}


// AdExSpikingNeurons Destructor
AdExSpikingNeurons::~AdExSpikingNeurons() {

}


int AdExSpikingNeurons::AddGroup(neuron_parameters_struct * group_params){

	int new_group_id = SpikingNeurons::AddGroup(group_params);

	AdEx_spiking_neuron_parameters_struct * AdEx_spiking_group_params = (AdEx_spiking_neuron_parameters_struct*)group_params;

	adaptation_values_w = (float*)realloc(adaptation_values_w, total_number_of_neurons*sizeof(float));
	membrane_capacitances_Cm = (float*)realloc(membrane_capacitances_Cm, total_number_of_neurons*sizeof(float));
	membrane_leakage_conductances_g0 = (float*)realloc(membrane_leakage_conductances_g0, total_number_of_neurons*sizeof(float));
	leak_reversal_potentials_E_L = (float*)realloc(leak_reversal_potentials_E_L, total_number_of_neurons*sizeof(float));
	slope_factors_Delta_T = (float*)realloc(slope_factors_Delta_T, total_number_of_neurons*sizeof(float));
	adaptation_coupling_coefficients_a = (float*)realloc(adaptation_coupling_coefficients_a, total_number_of_neurons*sizeof(float));
	adaptation_time_constants_tau_w = (float*)realloc(adaptation_time_constants_tau_w, total_number_of_neurons*sizeof(float));
	adaptation_changes_b = (float*)realloc(adaptation_changes_b, total_number_of_neurons*sizeof(float));

	absolute_refractory_period = AdEx_spiking_group_params->absolute_refractory_period;
	background_current = AdEx_spiking_group_params->background_current;


	for (int i = total_number_of_neurons - number_of_neurons_in_new_group; i < total_number_of_neurons; i++) {
		adaptation_values_w[i] = 0.0f;
		membrane_capacitances_Cm[i] = AdEx_spiking_group_params->membrane_capacitance_Cm;
		membrane_leakage_conductances_g0[i] = AdEx_spiking_group_params->membrane_leakage_conductance_g0;
		leak_reversal_potentials_E_L[i] = AdEx_spiking_group_params->leak_reversal_potential_E_L;
		slope_factors_Delta_T[i] = AdEx_spiking_group_params->slope_factor_Delta_T;
		adaptation_coupling_coefficients_a[i] = AdEx_spiking_group_params->adaptation_coupling_coefficient_a;
		adaptation_time_constants_tau_w[i] = AdEx_spiking_group_params->adaptation_time_constant_tau_w;
		adaptation_changes_b[i] = AdEx_spiking_group_params->adaptation_change_b;
	}

	return new_group_id;
}


void AdExSpikingNeurons::allocate_device_pointers(int maximum_axonal_delay_in_timesteps, bool high_fidelity_spike_storage) {

	SpikingNeurons::allocate_device_pointers(maximum_axonal_delay_in_timesteps, high_fidelity_spike_storage);

	CudaSafeCall(cudaMalloc((void **)&d_adaptation_values_w, sizeof(float)*total_number_of_neurons));
	CudaSafeCall(cudaMalloc((void **)&d_membrane_capacitances_Cm, sizeof(float)*total_number_of_neurons));
	CudaSafeCall(cudaMalloc((void **)&d_membrane_leakage_conductances_g0, sizeof(float)*total_number_of_neurons));
	CudaSafeCall(cudaMalloc((void **)&d_leak_reversal_potentials_E_L, sizeof(float)*total_number_of_neurons));
	CudaSafeCall(cudaMalloc((void **)&d_slope_factors_Delta_T, sizeof(float)*total_number_of_neurons));
	CudaSafeCall(cudaMalloc((void **)&d_adaptation_coupling_coefficients_a, sizeof(float)*total_number_of_neurons));
	CudaSafeCall(cudaMalloc((void **)&d_adaptation_time_constants_tau_w, sizeof(float)*total_number_of_neurons));
	CudaSafeCall(cudaMalloc((void **)&d_adaptation_changes_b, sizeof(float)*total_number_of_neurons));
}


void AdExSpikingNeurons::copy_constants_to_device() {

	SpikingNeurons::copy_constants_to_device();

	CudaSafeCall(cudaMemcpy(d_adaptation_values_w, adaptation_values_w, sizeof(float)*total_number_of_neurons, cudaMemcpyHostToDevice));
	CudaSafeCall(cudaMemcpy(d_membrane_capacitances_Cm, membrane_capacitances_Cm, sizeof(float)*total_number_of_neurons, cudaMemcpyHostToDevice));
	CudaSafeCall(cudaMemcpy(d_membrane_leakage_conductances_g0, membrane_leakage_conductances_g0, sizeof(float)*total_number_of_neurons, cudaMemcpyHostToDevice));
	CudaSafeCall(cudaMemcpy(d_leak_reversal_potentials_E_L, leak_reversal_potentials_E_L, sizeof(float)*total_number_of_neurons, cudaMemcpyHostToDevice));
	CudaSafeCall(cudaMemcpy(d_slope_factors_Delta_T, slope_factors_Delta_T, sizeof(float)*total_number_of_neurons, cudaMemcpyHostToDevice));
	CudaSafeCall(cudaMemcpy(d_adaptation_coupling_coefficients_a, adaptation_coupling_coefficients_a, sizeof(float)*total_number_of_neurons, cudaMemcpyHostToDevice));
	CudaSafeCall(cudaMemcpy(d_adaptation_time_constants_tau_w, adaptation_time_constants_tau_w, sizeof(float)*total_number_of_neurons, cudaMemcpyHostToDevice));
	CudaSafeCall(cudaMemcpy(d_adaptation_changes_b, d_adaptation_changes_b, sizeof(float)*total_number_of_neurons, cudaMemcpyHostToDevice));
}

void AdExSpikingNeurons::reset_neuron_activities() {

	SpikingNeurons::reset_neuron_activities();

	// Set adapatation value to zero
	CudaSafeCall(cudaMemcpy(d_adaptation_values_w, adaptation_values_w, sizeof(float)*total_number_of_neurons, cudaMemcpyHostToDevice));
}



void AdExSpikingNeurons::update_membrane_potentials(float timestep, float current_time_in_seconds) {

	AdEx_update_membrane_potentials<<<number_of_neuron_blocks_per_grid, threads_per_block>>>(
																	d_membrane_potentials_v,
																	d_adaptation_values_w,
																	d_adaptation_changes_b,
																	d_membrane_capacitances_Cm,
																	d_membrane_leakage_conductances_g0,
																	d_leak_reversal_potentials_E_L,
																	d_slope_factors_Delta_T,
																	d_adaptation_coupling_coefficients_a,
																	d_adaptation_time_constants_tau_w,
																	d_current_injections,
																	d_thresholds_for_action_potential_spikes,
																	d_last_spike_time_of_each_neuron,
																	absolute_refractory_period,
																	background_current,
																	current_time_in_seconds,
																	timestep,
																	total_number_of_neurons);

	CudaCheckError();
}


__global__ void AdEx_update_membrane_potentials(float *d_membrane_potentials_v,
								float * d_adaptation_values_w,
								float * d_adaptation_changes_b,
								float * d_membrane_capacitances_Cm,
								float * d_membrane_leakage_conductances_g0,
								float * d_leak_reversal_potentials_E_L,
								float * d_slope_factors_Delta_T,
								float * d_adaptation_coupling_coefficients_a,
								float * d_adaptation_time_constants_tau_w,
								float * d_current_injections,
								float * d_thresholds_for_action_potential_spikes,
								float * d_last_spike_time_of_each_neuron,
								float absolute_refractory_period,
								float background_current,
								float current_time_in_seconds,
								float timestep,
								size_t total_number_of_neurons){


	// // Get thread IDs
	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	while (idx < total_number_of_neurons) {
		// Check for refractory period:
		if ((current_time_in_seconds - d_last_spike_time_of_each_neuron[idx]) >= absolute_refractory_period){

			// Updating the membrane potential
			float inverse_capacitance = (1.0f / d_membrane_capacitances_Cm[idx]);
			float membrane_leak_diff = (d_membrane_potentials_v[idx] - d_leak_reversal_potentials_E_L[idx]);
			float membrane_leakage = -1.0 * d_membrane_leakage_conductances_g0[idx]*membrane_leak_diff;
			float membrane_thresh_diff = (d_membrane_potentials_v[idx] - d_thresholds_for_action_potential_spikes[idx]);

			// Checking for limit of Delta_T => 0
			float slope_adaptation = 0.0f;
			if (d_slope_factors_Delta_T[idx] != 0.0f){
				slope_adaptation = d_membrane_leakage_conductances_g0[idx]*d_slope_factors_Delta_T[idx]*expf(membrane_thresh_diff / d_slope_factors_Delta_T[idx]);
			}

			float update_membrane_potential = inverse_capacitance*(membrane_leakage + slope_adaptation - d_adaptation_values_w[idx] + d_current_injections[idx] + background_current);

			// Updating the adaptation parameter
			float inverse_tau_w = (1.0f / d_adaptation_time_constants_tau_w[idx]);
			float adaptation_change = d_adaptation_coupling_coefficients_a[idx]*membrane_leak_diff;

			float update_adaptation_value = inverse_tau_w*(adaptation_change - d_adaptation_values_w[idx]);


			//
			d_adaptation_values_w[idx] += timestep*update_adaptation_value;
			d_membrane_potentials_v[idx] += timestep*update_membrane_potential;

		} else {
			d_current_injections[idx] = 0.0f;
		}
		idx += blockDim.x * gridDim.x;

	}
	__syncthreads();
}


void AdExSpikingNeurons::check_for_neuron_spikes(float current_time_in_seconds, float timestep) {

	check_for_neuron_spikes_kernel<<<number_of_neuron_blocks_per_grid, threads_per_block>>>(d_membrane_potentials_v,
																	d_adaptation_values_w,
																	d_adaptation_changes_b,
																	d_thresholds_for_action_potential_spikes,
																	d_resting_potentials,
																	d_last_spike_time_of_each_neuron,
																	d_bitarray_of_neuron_spikes,
																	bitarray_length,
																	bitarray_maximum_axonal_delay_in_timesteps,
																	current_time_in_seconds,
																	timestep,
																	total_number_of_neurons,
																	high_fidelity_spike_flag);

	CudaCheckError();
}


// Spiking Neurons
__global__ void check_for_neuron_spikes_kernel(float *d_membrane_potentials_v,
								float *d_adaptation_values_w,
								float *d_adaptation_changes_b,
								float *d_thresholds_for_action_potential_spikes,
								float *d_resting_potentials,
								float* d_last_spike_time_of_each_neuron,
								unsigned char* d_bitarray_of_neuron_spikes,
								int bitarray_length,
								int bitarray_maximum_axonal_delay_in_timesteps,
								float current_time_in_seconds,
								float timestep,
								size_t total_number_of_neurons,
								bool high_fidelity_spike_flag) {

	// Get thread IDs
	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	while (idx < total_number_of_neurons) {
		if (d_membrane_potentials_v[idx] >= d_thresholds_for_action_potential_spikes[idx]) {

			// Set current time as last spike time of neuron
			d_last_spike_time_of_each_neuron[idx] = current_time_in_seconds;

			// Reset membrane potential
			d_membrane_potentials_v[idx] = d_resting_potentials[idx];

			// Set the adaptation parameter (w += b)
			d_adaptation_values_w[idx] += d_adaptation_changes_b[idx];

			// High fidelity spike storage
			if (high_fidelity_spike_flag){
				// Get start of the given neuron's bits
				int neuron_id_spike_store_start = idx * bitarray_length;
				// Get offset depending upon the current timestep
				int offset_index = (int)(round((float)(current_time_in_seconds / timestep))) % bitarray_maximum_axonal_delay_in_timesteps;
				int offset_byte = offset_index / 8;
				int offset_bit_pos = offset_index - (8 * offset_byte);
				// Get the specific position at which we should be putting the current value
				unsigned char byte = d_bitarray_of_neuron_spikes[neuron_id_spike_store_start + offset_byte];
				// Set the specific bit in the byte to on
				byte |= (1 << offset_bit_pos);
				// Assign the byte
				d_bitarray_of_neuron_spikes[neuron_id_spike_store_start + offset_byte] = byte;
			}

		} else {
			// High fidelity spike storage
			if (high_fidelity_spike_flag){
				// Get start of the given neuron's bits
				int neuron_id_spike_store_start = idx * bitarray_length;
				// Get offset depending upon the current timestep
				int offset_index = (int)(round((float)(current_time_in_seconds / timestep))) % bitarray_maximum_axonal_delay_in_timesteps;
				int offset_byte = offset_index / 8;
				int offset_bit_pos = offset_index - (8 * offset_byte);
				// Get the specific position at which we should be putting the current value
				unsigned char byte = d_bitarray_of_neuron_spikes[neuron_id_spike_store_start + offset_byte];
				// Set the specific bit in the byte to on
				byte &= ~(1 << offset_bit_pos);
				// Assign the byte
				d_bitarray_of_neuron_spikes[neuron_id_spike_store_start + offset_byte] = byte;
			}
		}

		idx += blockDim.x * gridDim.x;
	}
	__syncthreads();

}
