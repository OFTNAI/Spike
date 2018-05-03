#pragma once

#include "Spike/Synapses/ConductanceSpikingSynapses.hpp"
#include "SpikingSynapses.hpp"
#include "Spike/Backend/CUDA/CUDABackend.hpp"
#include <cuda.h>
#include <vector_types.h>
#include <curand.h>
#include <curand_kernel.h>

namespace Backend {
  namespace CUDA {
    class ConductanceSpikingSynapses : public virtual ::Backend::CUDA::SpikingSynapses,
                                       public virtual ::Backend::ConductanceSpikingSynapses {
    public:

      ~ConductanceSpikingSynapses() override;
      SPIKE_MAKE_BACKEND_CONSTRUCTOR(ConductanceSpikingSynapses);
      using ::Backend::ConductanceSpikingSynapses::frontend;

      // Variables used for memory-trace based synaptic input
      int conductance_trace_length = 0;
      float* neuron_wise_conductance_trace = nullptr;
      float* h_neuron_wise_conductance_trace = nullptr;
      float* d_decay_terms_tau_g = nullptr;
      float* d_reversal_potentials_Vhat = nullptr;

      void prepare() override;
      void reset_state() override;

      void allocate_device_pointers(); // Not virtual
      void copy_constants_and_initial_efficacies_to_device(); // Not virtual

      void state_update(::SpikingNeurons * neurons, ::SpikingNeurons * input_neurons, float current_time_in_seconds, float timestep) final;

    };

    __global__ void conductance_calculate_postsynaptic_current_injection_kernel(
      float* decay_term_values,
      float* reversal_values,
      int num_decay_terms,
      float* neuron_wise_conductance_traces,
      float* neuron_wise_input_update,
      float* d_neurons_current_injections,
      float* d_total_current_conductance,
      float timestep,
      int timestep_grouping,
      size_t total_number_of_neurons);
  }
}
