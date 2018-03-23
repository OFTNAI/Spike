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
      void apply_stdp_to_synapse_weights(float current_time_in_seconds, float timestep) override;
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
	   int timestep_grouping,
           float current_time_in_seconds,
           int* d_plastic_synapse_indices,
           size_t total_number_of_plastic_synapses);

  }
}
