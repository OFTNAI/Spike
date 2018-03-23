// -*- mode: c++ -*-
#include "Spike/Backend/CUDA/Neurons/AdExSpikingNeurons.hpp"

SPIKE_EXPORT_BACKEND_TYPE(CUDA, AdExSpikingNeurons);

namespace Backend {
  namespace CUDA {
    AdExSpikingNeurons::~AdExSpikingNeurons() {
      CudaSafeCall(cudaFree(adaptation_values_w));
      CudaSafeCall(cudaFree(membrane_capacitances_Cm));
      CudaSafeCall(cudaFree(membrane_leakage_conductances_g0));
      CudaSafeCall(cudaFree(leak_reversal_potentials_E_L));
      CudaSafeCall(cudaFree(slope_factors_Delta_T));
      CudaSafeCall(cudaFree(adaptation_coupling_coefficients_a));
      CudaSafeCall(cudaFree(adaptation_time_constants_tau_w));
      CudaSafeCall(cudaFree(adaptation_changes_b));
    }

    void AdExSpikingNeurons::allocate_device_pointers() {
      CudaSafeCall(cudaMalloc((void **)&adaptation_values_w, sizeof(float)*frontend()->total_number_of_neurons));
      CudaSafeCall(cudaMalloc((void **)&membrane_capacitances_Cm, sizeof(float)*frontend()->total_number_of_neurons));
      CudaSafeCall(cudaMalloc((void **)&membrane_leakage_conductances_g0, sizeof(float)*frontend()->total_number_of_neurons));
      CudaSafeCall(cudaMalloc((void **)&leak_reversal_potentials_E_L, sizeof(float)*frontend()->total_number_of_neurons));
      CudaSafeCall(cudaMalloc((void **)&slope_factors_Delta_T, sizeof(float)*frontend()->total_number_of_neurons));
      CudaSafeCall(cudaMalloc((void **)&adaptation_coupling_coefficients_a, sizeof(float)*frontend()->total_number_of_neurons));
      CudaSafeCall(cudaMalloc((void **)&adaptation_time_constants_tau_w, sizeof(float)*frontend()->total_number_of_neurons));
      CudaSafeCall(cudaMalloc((void **)&adaptation_changes_b, sizeof(float)*frontend()->total_number_of_neurons));
    }


    void AdExSpikingNeurons::copy_constants_to_device() {
      CudaSafeCall(cudaMemcpy(adaptation_values_w,
                              frontend()->adaptation_values_w,
                              sizeof(float)*frontend()->total_number_of_neurons,
                              cudaMemcpyHostToDevice));
      CudaSafeCall(cudaMemcpy(membrane_capacitances_Cm,
                              frontend()->membrane_capacitances_Cm,
                              sizeof(float)*frontend()->total_number_of_neurons,
                              cudaMemcpyHostToDevice));
      CudaSafeCall(cudaMemcpy(membrane_leakage_conductances_g0,
                              frontend()->membrane_leakage_conductances_g0,
                              sizeof(float)*frontend()->total_number_of_neurons,
                              cudaMemcpyHostToDevice));
      CudaSafeCall(cudaMemcpy(leak_reversal_potentials_E_L,
                              frontend()->leak_reversal_potentials_E_L,
                              sizeof(float)*frontend()->total_number_of_neurons,
                              cudaMemcpyHostToDevice));
      CudaSafeCall(cudaMemcpy(slope_factors_Delta_T,
                              frontend()->slope_factors_Delta_T,
                              sizeof(float)*frontend()->total_number_of_neurons,
                              cudaMemcpyHostToDevice));
      CudaSafeCall(cudaMemcpy(adaptation_coupling_coefficients_a,
                              frontend()->adaptation_coupling_coefficients_a,
                              sizeof(float)*frontend()->total_number_of_neurons,
                              cudaMemcpyHostToDevice));
      CudaSafeCall(cudaMemcpy(adaptation_time_constants_tau_w,
                              frontend()->adaptation_time_constants_tau_w,
                              sizeof(float)*frontend()->total_number_of_neurons,
                              cudaMemcpyHostToDevice));
      CudaSafeCall(cudaMemcpy(adaptation_changes_b,
                              frontend()->adaptation_changes_b,
                              sizeof(float)*frontend()->total_number_of_neurons,
                              cudaMemcpyHostToDevice));
    }

    void AdExSpikingNeurons::prepare() {
      SpikingNeurons::prepare();
      allocate_device_pointers();
      copy_constants_to_device();
    }

    void AdExSpikingNeurons::reset_state() {
      SpikingNeurons::reset_state();
      // Set adapatation value to zero
      CudaSafeCall(cudaMemcpy(adaptation_values_w,
                              frontend()->adaptation_values_w,
                              sizeof(float)*frontend()->total_number_of_neurons,
                              cudaMemcpyHostToDevice));
    }

    void AdExSpikingNeurons::state_update(float current_time_in_seconds, float timestep) {
	AdEx_update_membrane_potentials<<<number_of_neuron_blocks_per_grid, threads_per_block>>>
          (membrane_potentials_v,
           adaptation_values_w,
           adaptation_changes_b,
           membrane_capacitances_Cm,
           membrane_leakage_conductances_g0,
           leak_reversal_potentials_E_L,
           slope_factors_Delta_T,
           adaptation_coupling_coefficients_a,
           adaptation_time_constants_tau_w,
           current_injections,
           thresholds_for_action_potential_spikes,
	   resting_potentials,
           last_spike_time_of_each_neuron,
           frontend()->absolute_refractory_period,
           frontend()->background_current,
           current_time_in_seconds,
           timestep,
           frontend()->total_number_of_neurons);
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
						    float * d_resting_potentials,
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

          d_adaptation_values_w[idx] += timestep*update_adaptation_value;
          d_membrane_potentials_v[idx] += timestep*update_membrane_potential;

          if (d_membrane_potentials_v[idx] >= d_thresholds_for_action_potential_spikes[idx]) {
  
            // Set current time as last spike time of neuron
            d_last_spike_time_of_each_neuron[idx] = current_time_in_seconds;
  
            // Reset membrane potential
            d_membrane_potentials_v[idx] = d_resting_potentials[idx];
  
            // Set the adaptation parameter (w += b)
            d_adaptation_values_w[idx] += d_adaptation_changes_b[idx]; 
          }

        }
        idx += blockDim.x * gridDim.x;

      }
      __syncthreads();
    }

  }
}
