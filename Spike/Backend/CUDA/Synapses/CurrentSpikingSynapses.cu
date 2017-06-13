// -*- mode: c++ -*-
#include "Spike/Backend/CUDA/Synapses/CurrentSpikingSynapses.hpp"

SPIKE_EXPORT_BACKEND_TYPE(CUDA, CurrentSpikingSynapses);

namespace Backend {
  namespace CUDA {
    void CurrentSpikingSynapses::prepare() {
      SpikingSynapses::prepare();
    }

    void CurrentSpikingSynapses::reset_state() {
      SpikingSynapses::reset_state();
    }

    void CurrentSpikingSynapses::state_update(::SpikingNeurons * neurons, ::SpikingNeurons* input_neurons, float current_time_in_seconds, float timestep) {
      ::Backend::CUDA::SpikingNeurons* neurons_backend
          = dynamic_cast<::Backend::CUDA::SpikingNeurons*>(neurons->backend());
      current_calculate_postsynaptic_current_injection_kernel<<<number_of_synapse_blocks_per_grid, threads_per_block>>>(
         synaptic_efficacies_or_weights,
         time_of_last_spike_to_reach_synapse,
         postsynaptic_neuron_indices,
         neurons_backend->current_injections,
         current_time_in_seconds,
         frontend()->total_number_of_synapses);

      CudaCheckError();
    }

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
  
