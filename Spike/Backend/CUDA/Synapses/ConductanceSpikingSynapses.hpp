#pragma once

#include "Spike/Synapses/ConductanceSpikingSynapses.hpp"
#include "Spike/Backend/CUDA/CUDABackend.hpp"
#include <cuda.h>
#include <vector_types.h>
#include <curand.h>
#include <curand_kernel.h>

namespace Backend {
  namespace CUDA {
    class ConductanceSpikingSynapses : public ::Backend::ConductanceSpikingSynapses {
    public:
      float * synaptic_conductances_g = NULL;
      float * biological_conductance_scaling_constants_lambda = NULL;
      float * reversal_potentials_Vhat = NULL;
      float * decay_terms_tau_g = NULL;

      virtual void allocate_device_pointers();
      virtual void copy_constants_and_initial_efficacies_to_device();
      virtual void reset_state();
    };

    __global__ void conductance_calculate_postsynaptic_current_injection_kernel
    (int * d_presynaptic_neuron_indices,
     int* d_postsynaptic_neuron_indices,
     float* d_reversal_potentials_Vhat,
     float* d_neurons_current_injections,
     size_t total_number_of_synapses,
     float * d_membrane_potentials_v,
     float * d_synaptic_conductances_g);

    __global__ void conductance_update_synaptic_conductances_kernel
    (float timestep, 
     float * d_synaptic_conductances_g, 
     float * d_synaptic_efficacies_or_weights, 
     float * d_time_of_last_spike_to_reach_synapse,
     float * d_biological_conductance_scaling_constants_lambda,
     int total_number_of_synapses,
     float current_time_in_seconds,
     float * d_decay_terms_tau_g);
  }
}
