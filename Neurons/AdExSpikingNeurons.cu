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

	d_adaptation_values_w = NULL;
	d_membrane_capacitances_Cm = NULL;
	d_membrane_leakage_conductances_g0 = NULL;
	d_leak_reversal_potentials_E_L = NULL;
	d_slope_factors_Delta_T = NULL;
	d_adaptation_coupling_coefficients_a = NULL;
	d_adaptation_time_constants_tau_w = NULL;
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

	
	for (int i = total_number_of_neurons - number_of_neurons_in_new_group; i < total_number_of_neurons; i++) {
		adaptation_values_w[i] = 0.0f;
		membrane_capacitances_Cm[i] = AdEx_spiking_group_params->membrane_capacitance_Cm;
		membrane_leakage_conductances_g0[i] = AdEx_spiking_group_params->membrane_leakage_conductance_g0;
		leak_reversal_potentials_E_L[i] = AdEx_spiking_group_params->leak_reversal_potential_E_L;
		slope_factors_Delta_T[i] = AdEx_spiking_group_params->slope_factor_Delta_T;
		adaptation_coupling_coefficients_a[i] = AdEx_spiking_group_params->adaptation_coupling_coefficient_a;
		adaptation_time_constants_tau_w[i] = AdEx_spiking_group_params->adaptation_time_constant_tau_w;
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
}

void AdExSpikingNeurons::reset_neuron_activities() {

	SpikingNeurons::reset_neuron_activities();

	// Set adapatation value to zero
	CudaSafeCall(cudaMemcpy(d_adaptation_values_w, adaptation_values_w, sizeof(float)*total_number_of_neurons, cudaMemcpyHostToDevice));
}



void AdExSpikingNeurons::update_membrane_potentials(float timestep) {

	AdEx_update_membrane_potentials<<<number_of_neuron_blocks_per_grid, threads_per_block>>>(
																	d_membrane_potentials_v,
																	d_adaptation_values_w,
																	d_membrane_capacitances_Cm,
																	d_membrane_leakage_conductances_g0,
																	d_leak_reversal_potentials_E_L,
																	d_slope_factors_Delta_T,
																	d_adaptation_coupling_coefficients_a,
																	d_adaptation_time_constants_tau_w,
																	d_current_injections,
																	d_thresholds_for_action_potential_spikes,
																	timestep,
																	total_number_of_neurons);

	CudaCheckError();
}


__global__ void AdEx_update_membrane_potentials(float *d_membrane_potentials_v,
								float * d_adaptation_values_w,
								float * d_membrane_capacitances_Cm,
								float * d_membrane_leakage_conductances_g0,
								float * d_leak_reversal_potentials_E_L,
								float * d_slope_factors_Delta_T,
								float * d_adaptation_coupling_coefficients_a,
								float * d_adaptation_time_constants_tau_w,
								float * d_current_injections,
								float * d_thresholds_for_action_potential_spikes,
								float timestep,
								size_t total_number_of_neurons){

	
	// // Get thread IDs
	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	while (idx < total_number_of_neurons) {

		// Updating the membrane potential
		float inverse_capacitance = (1 / d_membrane_capacitances_Cm[idx]);
		float membrane_leak_diff = (d_membrane_potentials_v[idx] - d_leak_reversal_potentials_E_L[idx]);
		float membrane_leakage = - d_membrane_leakage_conductances_g0[idx]*membrane_leak_diff;
		float membrane_thresh_diff = (d_membrane_potentials_v[idx] - d_thresholds_for_action_potential_spikes[idx]);
		float slope_adaptation = d_membrane_leakage_conductances_g0[idx]*d_slope_factors_Delta_T[idx]*expf(membrane_thresh_diff / d_slope_factors_Delta_T[idx]);

		float new_membrane_potential = inverse_capacitance*(membrane_leakage + slope_adaptation - d_adaptation_values_w[idx] + d_current_injections[idx]);

		// Updating the adaptation parameter
		float inverse_tau_w = (1 / d_adaptation_time_constants_tau_w[idx]);
		float adaptation_change = d_adaptation_coupling_coefficients_a[idx]*membrane_leak_diff;

		float new_adaptation_value = inverse_tau_w*(adaptation_change - d_adaptation_values_w[idx]);


		// 
		d_adaptation_values_w[idx] = new_adaptation_value;
		d_membrane_potentials_v[idx] = new_membrane_potential;

		idx += blockDim.x * gridDim.x;

	}
	__syncthreads();
}


