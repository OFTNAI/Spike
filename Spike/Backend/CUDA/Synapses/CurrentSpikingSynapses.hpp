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

      void calculate_postsynaptic_current_injection
      (::SpikingNeurons * neurons,
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

    __global__ void current_apply_ltd_to_synapse_weights_kernel
    (float* d_time_of_last_spike_to_reach_synapse,
     float* d_synaptic_efficacies_or_weights,
     bool* d_stdp,
     float* d_last_spike_time_of_each_neuron,
     int* d_postsyns,
     float currtime,
     struct stdp_struct stdp_vars,
     size_t total_number_of_synapse);

    __global__ void current_apply_ltp_to_synapse_weights_kernel
    (int* d_postsyns,
     float* d_last_spike_time_of_each_neuron,
     bool* d_stdp,
     float* d_time_of_last_spike_to_reach_synapse,
     float* d_synaptic_efficacies_or_weights,
     struct stdp_struct stdp_vars,
     float currtime,
     size_t total_number_of_synapse);
  }
}

