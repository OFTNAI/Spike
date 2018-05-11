#pragma once

#include "Spike/Plasticity/WeightDependentSTDPPlasticity.hpp"
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
    class WeightDependentSTDPPlasticity : public virtual ::Backend::CUDA::STDPPlasticity,
                           public virtual ::Backend::WeightDependentSTDPPlasticity {
    public:
      float* stdp_pre_memory_trace = nullptr;
      float* stdp_post_memory_trace = nullptr;
      float* h_stdp_trace = nullptr;

      ~WeightDependentSTDPPlasticity() override;
      SPIKE_MAKE_BACKEND_CONSTRUCTOR(WeightDependentSTDPPlasticity);
      using ::Backend::WeightDependentSTDPPlasticity::frontend;

      void prepare() override;
      void reset_state() override;

      void allocate_device_pointers();
      void apply_stdp_to_synapse_weights(float current_time_in_seconds, float timestep) override;
    };

    __global__ void ltp_and_ltd
          (int* d_postsyns,
           float* d_time_of_last_spike_to_reach_synapse,
           float* d_last_spike_time_of_each_neuron,
           float* d_synaptic_efficacies_or_weights,
           float* stdp_pre_memory_trace,
           float* stdp_post_memory_trace,
           struct weightdependent_stdp_plasticity_parameters_struct stdp_vars,
           float timestep,
           int timestep_grouping,
           float current_time_in_seconds,
           int* d_plastic_synapse_indices,
           size_t total_number_of_plastic_synapses);

  }
}
