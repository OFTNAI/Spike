#pragma once

#include "Spike/STDP/MasquelierSTDP.hpp"
#include "STDP.hpp"
#include "Spike/Backend/CUDA/CUDABackend.hpp"
#include <cuda.h>
#include <vector_types.h>
#include <curand.h>
#include <curand_kernel.h>

namespace Backend {
  namespace CUDA {
    class MasquelierSTDP : public virtual ::Backend::CUDA::STDPCommon,
                           public ::Backend::MasquelierSTDP {
    public:
      int* index_of_last_afferent_synapse_to_spike = nullptr;
      bool* isindexed_ltd_synapse_spike = nullptr;
      int* index_of_first_synapse_spiked_after_postneuron = nullptr;

      MAKE_BACKEND_CONSTRUCTOR(MasquelierSTDP);

      virtual void allocate_device_pointers();
      virtual void prepare() {
        allocate_device_pointers();
      }
      virtual void reset_state();
      virtual void apply_stdp_to_synapse_weights(float* d_last_spike_time_of_each_neuron, float current_time_in_seconds);
    };

    // Kernel to carry out LTP/LTD
    __global__ void apply_stdp_to_synapse_weights_kernel
    (int* d_postsyns,
     float* d_last_spike_time_of_each_neuron,
     bool* d_stdp,
     float* d_time_of_last_spike_to_reach_synapse,
     float* d_synaptic_efficacies_or_weights,
     int* d_index_of_last_afferent_synapse_to_spike,
     bool* d_isindexed_ltd_synapse_spike,
     int* d_index_of_first_synapse_spiked_after_postneuron,
     struct masquelier_stdp_parameters_struct stdp_vars,
     float currtime,
     size_t total_number_of_post_neurons);

    __global__ void get_indices_to_apply_stdp
    (int* d_postsyns,
     float* d_last_spike_time_of_each_neuron,
     bool* d_stdp,
     float* d_time_of_last_spike_to_reach_synapse,
     int* d_index_of_last_afferent_synapse_to_spike,
     bool* d_isindexed_ltd_synapse_spike,
     int* d_index_of_first_synapse_spiked_after_postneuron,
     float currtime,
     size_t total_number_of_synapse);
  }
}
