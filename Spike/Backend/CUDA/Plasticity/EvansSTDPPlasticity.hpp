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
      void update_synaptic_efficacies_or_weights(float current_time_in_seconds, float timestep) override;
    };
    
    __global__ void ltp_and_ltd
          (int* d_postsyns,
           int* d_presyns,
           int* d_syndelays,
           spiking_neurons_data_struct* neuron_data,
           spiking_neurons_data_struct* input_neuron_data,
           float* d_synaptic_efficacies_or_weights,
           float* recent_presynaptic_activities_C,
           float* recent_postsynaptic_activities_D,
           evans_stdp_plasticity_parameters_struct stdp_vars,
           float timestep,
           int timestep_grouping,
           float current_time_in_seconds,
           int* d_plastic_synapse_indices,
           size_t total_number_of_plastic_synapses);

  }
}
