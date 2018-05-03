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

      neuron_data = new lif_spiking_neurons_data_struct();
      (spiking_neurons_data_struct)*neuron_data = *(static_cast<LIFSpikingNeurons*>(this)->SpikingNeurons::neuron_data);
      neuron_data->membrane_time_constants_tau_m = membrane_time_constants_tau_m;
      neuron_data->membrane_resistances_R = membrane_resistances_R;
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
	 total_current_conductance,
	 thresholds_for_action_potential_spikes,
         frontend()->background_current,
         timestep,
	 frontend()->model->timestep_grouping,
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
                                                   float* d_total_current_conductance,
						   float* d_threshold_for_action_potential_spikes,
                                                   float background_current,
                                                   float timestep,
						   int timestep_grouping,
                                                   float current_time_in_seconds,
                                                   float refractory_period_in_seconds,
                                                   size_t total_number_of_neurons) {
      // // Get thread IDs
      int idx = threadIdx.x + blockIdx.x * blockDim.x;
      while (idx < total_number_of_neurons) {

          float equation_constant = timestep / d_membrane_time_constants_tau_m[idx];
          float resting_potential_V0 = d_resting_potentials[idx];
          float temp_membrane_resistance_R = d_membrane_resistances_R[idx];
          float current_injection_Ii = d_current_injections[idx*timestep_grouping];
          float total_current_conductance = d_total_current_conductance[idx*timestep_grouping];
          float membrane_potential_Vi = d_membrane_potentials_v[idx];

  	  for (int g=0; g < timestep_grouping; g++){	  
            if (((current_time_in_seconds + g*timestep) - d_last_spike_time_of_each_neuron[idx]) >= refractory_period_in_seconds){
              current_injection_Ii = d_current_injections[g*total_number_of_neurons + idx];
              total_current_conductance = d_total_current_conductance[g*total_number_of_neurons + idx];
              float new_membrane_potential = equation_constant * (resting_potential_V0 + temp_membrane_resistance_R * (current_injection_Ii - total_current_conductance*membrane_potential_Vi)) + (1 - equation_constant) * membrane_potential_Vi + equation_constant * background_current;
	  
	      // Finally check for a spike
	      if (new_membrane_potential >= d_threshold_for_action_potential_spikes[idx]){
	  	  d_last_spike_time_of_each_neuron[idx] = current_time_in_seconds + (g*timestep);
		  membrane_potential_Vi = d_resting_potentials[idx];
		  break;
	      }

              membrane_potential_Vi = new_membrane_potential;
	    }
	  }
          
	  d_membrane_potentials_v[idx] = membrane_potential_Vi;
	  
          idx += blockDim.x * gridDim.x;
        }
     } 


  } // namespace CUDA
} // namespace Backend
