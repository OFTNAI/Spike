// -*- mode: c++ -*-
#include "Spike/Backend/CUDA/Neurons/LIFSpikingNeurons.hpp"

SPIKE_EXPORT_BACKEND_TYPE(CUDA, LIFSpikingNeurons);

namespace Backend {
  namespace CUDA {
    LIFSpikingNeurons::~LIFSpikingNeurons() {
      CudaSafeCall(cudaFree(membrane_time_constants_tau_m));
      CudaSafeCall(cudaFree(membrane_resistances_R));
    }

    void LIFSpikingNeurons::allocate_device_pointers() {
      CudaSafeCall(cudaMalloc((void **)&membrane_time_constants_tau_m, sizeof(float)*frontend()->total_number_of_neurons));
      CudaSafeCall(cudaMalloc((void **)&membrane_resistances_R, sizeof(float)*frontend()->total_number_of_neurons));
    }

    void LIFSpikingNeurons::copy_constants_to_device() {
      CudaSafeCall(cudaMemcpy(membrane_time_constants_tau_m,
                              frontend()->membrane_time_constants_tau_m,
                              sizeof(float)*frontend()->total_number_of_neurons,
                              cudaMemcpyHostToDevice));
      CudaSafeCall(cudaMemcpy(membrane_resistances_R,
                              frontend()->membrane_resistances_R,
                              sizeof(float)*frontend()->total_number_of_neurons,
                              cudaMemcpyHostToDevice));
    }

    void LIFSpikingNeurons::prepare() {
      SpikingNeurons::prepare();
      allocate_device_pointers();
      copy_constants_to_device();
    }

    void LIFSpikingNeurons::reset_state() {
      SpikingNeurons::reset_state();
    }

    void LIFSpikingNeurons::state_update(float current_time_in_seconds, float timestep) {
      lif_update_membrane_potentials<<<number_of_neuron_blocks_per_grid, threads_per_block>>>
        (membrane_potentials_v,
         last_spike_time_of_each_neuron,
         membrane_resistances_R,
         membrane_time_constants_tau_m,
         resting_potentials,
         current_injections,
	 thresholds_for_action_potential_spikes,
         frontend()->background_current,
         timestep,
         current_time_in_seconds,
         frontend()->refractory_period_in_seconds,
         frontend()->total_number_of_neurons);

      CudaCheckError();
    }
    
    __global__ void lif_update_membrane_potentials(float *d_membrane_potentials_v,
                                                   float * d_last_spike_time_of_each_neuron,
                                                   float * d_membrane_resistances_R,
                                                   float * d_membrane_time_constants_tau_m,
                                                   float * d_resting_potentials,
                                                   float* d_current_injections,
						   float* d_threshold_for_action_potential_spikes,
                                                   float background_current,
                                                   float timestep,
                                                   float current_time_in_seconds,
                                                   float refractory_period_in_seconds,
                                                   size_t total_number_of_neurons) {
      // // Get thread IDs
      int idx = threadIdx.x + blockIdx.x * blockDim.x;
      while (idx < total_number_of_neurons) {

        if ((current_time_in_seconds - d_last_spike_time_of_each_neuron[idx]) >= refractory_period_in_seconds){
          float equation_constant = timestep / d_membrane_time_constants_tau_m[idx];
          float membrane_potential_Vi = d_membrane_potentials_v[idx];
          float current_injection_Ii = d_current_injections[idx];
          float resting_potential_V0 = d_resting_potentials[idx];
          float temp_membrane_resistance_R = d_membrane_resistances_R[idx];
	
          float new_membrane_potential = equation_constant * (resting_potential_V0 + temp_membrane_resistance_R * current_injection_Ii) + (1 - equation_constant) * membrane_potential_Vi + equation_constant * background_current;
	  
	  // Finally check for a spike
	  if (new_membrane_potential >= d_threshold_for_action_potential_spikes[idx]){
	  	d_last_spike_time_of_each_neuron[idx] = current_time_in_seconds;
		new_membrane_potential = d_resting_potentials[idx];
	  }
          
	  d_membrane_potentials_v[idx] = new_membrane_potential;
        }

        idx += blockDim.x * gridDim.x;

      }
      __syncthreads();
    }

  } // namespace CUDA
} // namespace Backend
