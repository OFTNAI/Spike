#include "Spike/Backend/CUDA/Synapses/CurrentSpikingSynapses.hpp"

namespace Backend {
  namespace CUDA {

    __global__ void current_calculate_postsynaptic_current_injection_kernel
    (float* d_synaptic_efficacies_or_weights,
     float* d_time_of_last_spike_to_reach_synapse,
     int* d_postsynaptic_neuron_indices,
     float* d_neurons_current_injections,
     float current_time_in_seconds,
     size_t total_number_of_synapses){

      int idx = threadIdx.x + blockIdx.x * blockDim.x;
      if (idx < total_number_of_synapses) {

        if (d_time_of_last_spike_to_reach_synapse[idx] == current_time_in_seconds) {

          atomicAdd(&d_neurons_current_injections[d_postsynaptic_neuron_indices[idx]], d_synaptic_efficacies_or_weights[idx]);

        }
        idx += blockDim.x * gridDim.x;
      }
      __syncthreads();
    }

  }
}
  
