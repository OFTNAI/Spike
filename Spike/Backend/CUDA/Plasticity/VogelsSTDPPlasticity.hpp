#pragma once

#include "Spike/Plasticity/VogelsSTDPPlasticity.hpp"
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
    class VogelsSTDPPlasticity : public virtual ::Backend::CUDA::STDPPlasticity,
                           public virtual ::Backend::VogelsSTDPPlasticity {
    public:
      float* vogels_memory_trace_reset = nullptr; 
      float* vogels_pre_memory_trace = nullptr;
      float* vogels_post_memory_trace = nullptr;

      ~VogelsSTDPPlasticity() override;
      SPIKE_MAKE_BACKEND_CONSTRUCTOR(VogelsSTDPPlasticity);
      using ::Backend::VogelsSTDPPlasticity::frontend;

      void prepare() override;
      void reset_state() override;

      void allocate_device_pointers();
      void apply_stdp_to_synapse_weights(float current_time_in_seconds, float timestep) override;
    };

    // Kernel to carry out LTP/LTD
    __global__ void vogels_apply_stdp_to_synapse_weights_kernel
    (int* d_presyns,
     int* d_postsyns,
     float* d_last_spike_time_of_each_neuron,
     float* d_synaptic_efficacies_or_weights,
     float* vogels_pre_memory_trace,
     float* vogels_post_memory_trace,
     struct vogels_stdp_plasticity_parameters_struct stdp_vars,
     float currtime,
     float timestep,
     int* d_plastic_synapse_indices,
     size_t total_number_of_plastic_synapses);

  }
}
