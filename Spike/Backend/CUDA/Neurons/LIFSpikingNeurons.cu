#include "Spike/Backend/CUDA/Neurons/LIFSpikingNeurons.hpp"

namespace Backend {
  namespace CUDA {
    LIFSpikingNeurons::~LIFSpikingNeurons() {
      CudaSafeCall(cudaFree(d_membrane_time_constants_tau_m));
      CudaSafeCall(cudaFree(d_membrane_resistances_R));
    }

    void LIFSpikingNeurons::allocate_device_pointers(int maximum_axonal_delay_in_timesteps, bool high_fidelity_spike_storage) {
      SpikingNeurons::allocate_device_pointers(maximum_axonal_delay_in_timesteps, high_fidelity_spike_storage);

      CudaSafeCall(cudaMalloc((void **)&d_membrane_time_constants_tau_m, sizeof(float)*total_number_of_neurons));
      CudaSafeCall(cudaMalloc((void **)&d_membrane_resistances_R, sizeof(float)*total_number_of_neurons));
    }

    void LIFSpikingNeurons::copy_constants_to_device() {

      SpikingNeurons::copy_constants_to_device();

      CudaSafeCall(cudaMemcpy(d_membrane_time_constants_tau_m, membrane_time_constants_tau_m, sizeof(float)*total_number_of_neurons, cudaMemcpyHostToDevice));
      CudaSafeCall(cudaMemcpy(d_membrane_resistances_R, membrane_resistances_R, sizeof(float)*total_number_of_neurons, cudaMemcpyHostToDevice));
    }

    void LIFSpikingNeurons::update_membrane_potentials(float timestep, float current_time_in_seconds) {

      lif_update_membrane_potentials<<<number_of_neuron_blocks_per_grid, threads_per_block>>>
        (d_membrane_potentials_v,
         d_last_spike_time_of_each_neuron,
         d_membrane_resistances_R,
         d_membrane_time_constants_tau_m,
         d_resting_potentials,
         d_current_injections,
         timestep,
         current_time_in_seconds,
         refractory_period_in_seconds,
         total_number_of_neurons);

      CudaCheckError();
    }
    
    __global__ void lif_update_membrane_potentials(float *d_membrane_potentials_v,
                                                   float * d_last_spike_time_of_each_neuron,
                                                   float * d_membrane_resistances_R,
                                                   float * d_membrane_time_constants_tau_m,
                                                   float * d_resting_potentials,
                                                   float* d_current_injections,
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
	
          float new_membrane_potential = equation_constant * (resting_potential_V0 + temp_membrane_resistance_R * current_injection_Ii) + (1 - equation_constant) * membrane_potential_Vi;
          d_membrane_potentials_v[idx] = new_membrane_potential;
        }

        idx += blockDim.x * gridDim.x;

      }
      __syncthreads();
    }

  } // namespace CUDA
} // namespace Backend
