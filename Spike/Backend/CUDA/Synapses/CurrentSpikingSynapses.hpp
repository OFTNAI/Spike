#pragma once

#include "SpikingSynapses.hpp"

#include "Spike/Synapses/CurrentSpikingSynapses.hpp"
#include "Spike/Backend/CUDA/CUDABackend.hpp"
#include "Spike/Backend/CUDA/Neurons/SpikingNeurons.hpp"

#include <cuda.h>
#include <vector_types.h>
#include <curand.h>
#include <curand_kernel.h>

namespace Backend {
  namespace CUDA {
    class CurrentSpikingSynapses : public virtual ::Backend::CUDA::SpikingSynapses,
                                   public virtual ::Backend::CurrentSpikingSynapses {
    public:
      ~CurrentSpikingSynapses() override = default;
      SPIKE_MAKE_BACKEND_CONSTRUCTOR(CurrentSpikingSynapses);
      using ::Backend::CurrentSpikingSynapses::frontend;

      void prepare() override;
      void reset_state() override;

      void state_update
      (::SpikingNeurons * neurons,
       ::SpikingNeurons* input_neurons,
       float current_time_in_seconds,
       float timestep) final; // Overrides ::Backend::SpikingSynapses:: ...
    };

    __global__ void current_calculate_postsynaptic_current_injection_kernel
    (float* d_synaptic_efficacies_or_weights,
     float* d_time_of_last_spike_to_reach_synapse,
     int* d_postsynaptic_neuron_indices,
     float* d_neurons_current_injections,
     float current_time_in_seconds,
     size_t total_number_of_synapses);
  }
}

