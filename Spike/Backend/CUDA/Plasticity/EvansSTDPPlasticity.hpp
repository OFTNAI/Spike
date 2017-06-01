#pragma once

#include "Spike/Plasticity/EvansSTDPPlasticity.hpp"
#include "STDPPlasticity.hpp"
#include "Spike/Backend/CUDA/CUDABackend.hpp"
#include <cuda.h>
#include <vector_types.h>
#include <curand.h>
#include <curand_kernel.h>

namespace Backend {
  namespace CUDA {
    class EvansSTDPPlasticity : public virtual ::Backend::CUDA::STDPPlasticity,
                      public virtual ::Backend::EvansSTDPPlasticity {
    public:
      float* recent_postsynaptic_activities_D = nullptr; // (NEURON-WISE)
      float* recent_presynaptic_activities_C = nullptr;  // (SYNAPSE-WISE)

      ~EvansSTDPPlasticity() override;
      SPIKE_MAKE_BACKEND_CONSTRUCTOR(EvansSTDPPlasticity);
      using ::Backend::EvansSTDPPlasticity::frontend;

      void prepare() override;
      void reset_state() override;

      void allocate_device_pointers();
      void update_synaptic_efficacies_or_weights(float current_time_in_seconds) override;
      void update_presynaptic_activities(float timestep, float current_time_in_seconds) override;
      void update_postsynaptic_activities(float timestep, float current_time_in_seconds) override;
    };

    __global__ void update_postsynaptic_activities_kernel(float timestep,
								size_t total_number_of_neurons,
								float * d_recent_postsynaptic_activities_D,
								float * d_last_spike_time_of_each_neuron,
								float current_time_in_seconds,
								float decay_term_tau_D,
								float model_parameter_alpha_D);

    __global__ void update_presynaptic_activities_C_kernel
    (float* d_recent_presynaptic_activities_C,
     float* d_time_of_last_spike_to_reach_synapse,
     float timestep,
     float current_time_in_seconds,
     float synaptic_neurotransmitter_concentration_alpha_C,
     float decay_term_tau_C,
     int* d_plastic_synapse_indices,
     size_t total_number_of_plastic_synapses);

    __global__ void update_synaptic_efficacies_or_weights_kernel
    (float * d_recent_presynaptic_activities_C,
     float * d_recent_postsynaptic_activities_D,
     int* d_postsynaptic_neuron_indices,
     float* d_synaptic_efficacies_or_weights,
     float current_time_in_seconds,
     float * d_time_of_last_spike_to_reach_synapse,
     float * d_last_spike_time_of_each_neuron,
     float learning_rate_rho,
     int* d_plastic_synapse_indices,
     size_t total_number_of_plastic_synapses);

  }
}
