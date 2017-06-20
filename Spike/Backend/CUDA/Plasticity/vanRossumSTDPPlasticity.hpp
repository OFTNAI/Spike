#pragma once

#include "Spike/Plasticity/vanRossumSTDPPlasticity.hpp"
#include "STDPPlasticity.hpp"

#include "Spike/Backend/CUDA/Neurons/SpikingNeurons.hpp"
#include "Spike/Backend/CUDA/Synapses/SpikingSynapses.hpp"

#include "Spike/Backend/CUDA/CUDABackend.hpp"
#include <cuda.h>
#include <vector_types.h>
#include <curand.h>
#include <curand_kernel.h>

namespace Backend {
  namespace CUDA {
    class vanRossumSTDPPlasticity : public virtual ::Backend::CUDA::STDPPlasticity,
                           public virtual ::Backend::vanRossumSTDPPlasticity {
    public:
      int* index_of_last_afferent_synapse_to_spike = nullptr;
      bool* isindexed_ltd_synapse_spike = nullptr;
      int* index_of_first_synapse_spiked_after_postneuron = nullptr;
      float* stdp_pre_memory_trace = nullptr;
      float* stdp_post_memory_trace = nullptr;
      float* h_stdp_trace = nullptr;

      ~vanRossumSTDPPlasticity() override;
      SPIKE_MAKE_BACKEND_CONSTRUCTOR(vanRossumSTDPPlasticity);
      using ::Backend::vanRossumSTDPPlasticity::frontend;

      void prepare() override;
      void reset_state() override;

      void allocate_device_pointers();
      void apply_stdp_to_synapse_weights(float current_time_in_seconds) override;
    };

    __global__ void vanrossum_ltp_and_ltd
          (int* d_postsyns,
           float* d_time_of_last_spike_to_reach_synapse,
     	   float* d_last_spike_time_of_each_neuron,
           float* d_synaptic_efficacies_or_weights,
           float* stdp_pre_memory_trace,
           float* stdp_post_memory_trace,
           struct vanrossum_stdp_plasticity_parameters_struct stdp_vars,
           float timestep,
           float current_time_in_seconds,
           int* d_plastic_synapse_indices,
           size_t total_number_of_plastic_synapses);

    __global__ void vanrossum_apply_stdp_to_synapse_weights_kernel_nearest
    (int* d_postsyns,
     float* d_last_spike_time_of_each_neuron,
     float* d_time_of_last_spike_to_reach_synapse,
     float* d_synaptic_efficacies_or_weights,
     int* d_index_of_last_afferent_synapse_to_spike,
     bool* d_isindexed_ltd_synapse_spike,
     int* d_index_of_first_synapse_spiked_after_postneuron,
     struct vanrossum_stdp_plasticity_parameters_struct stdp_vars,
     float currtime,
     size_t total_number_of_post_neurons);

    __global__ void vanrossum_get_indices_to_apply_stdp
    (int* d_postsyns,
     float* d_last_spike_time_of_each_neuron,
     float* d_time_of_last_spike_to_reach_synapse,
     int* d_index_of_last_afferent_synapse_to_spike,
     bool* d_isindexed_ltd_synapse_spike,
     int* d_index_of_first_synapse_spiked_after_postneuron,
     float currtime,
     int* d_plastic_synapse_indices,
     size_t total_number_of_plastic_synapses);
  }
}
