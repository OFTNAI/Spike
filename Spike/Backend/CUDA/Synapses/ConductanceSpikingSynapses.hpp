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
      
      // Arrays listing synapse-wise properties
      float * biological_conductance_scaling_constants_lambda = nullptr;

      // Variables used to determine active/inactive synapses
      int* circular_spikenum_buffer = nullptr;
      int* spikeid_buffer = nullptr;
      int buffersize = 0;

      // Variables used for memory-trace based synaptic input
      int num_decay_terms = 0;
      float* decay_term_values = nullptr;
      float* h_decay_term_values = nullptr;
      float* reversal_values = nullptr;
      float* h_reversal_values = nullptr;
      int conductance_trace_length = 0;
      float* neuron_wise_conductance_trace = nullptr;
      float* h_neuron_wise_conductance_trace = nullptr;
      int* synapse_decay_id = nullptr;
      int* h_synapse_decay_id = nullptr;

      void prepare() override;
      void reset_state() override;

      void allocate_device_pointers(); // Not virtual
      void copy_constants_and_initial_efficacies_to_device(); // Not virtual

      void state_update(::SpikingNeurons * neurons, ::SpikingNeurons * input_neurons, float current_time_in_seconds, float timestep) final;

    };

    __global__ void get_active_synapses_kernel(
		int* d_per_neuron_efferent_synapse_count,
        	int* d_per_neuron_efferent_synapse_total,
                int* d_per_neuron_efferent_synapse_indices,
		int* d_per_input_neuron_efferent_synapse_count,
        	int* d_per_input_neuron_efferent_synapse_total,
                int* d_per_input_neuron_efferent_synapse_indices,
                int* d_delays,
                int* d_spikes_travelling_to_synapse,
                float* d_last_spike_time_of_each_neuron,
                float* d_last_spike_time_of_each_input_neuron,
                float current_time_in_seconds,
		int* circular_spikenum_buffer,
		int* spikeid_buffer,
		int bufferloc,
		int buffersize,
		int total_number_of_synapses,
                float timestep,
		int number_of_input_neurons,
                size_t total_number_of_neurons); 

    __global__ void conductance_calculate_postsynaptic_current_injection_kernel(
      float* decay_term_values,
      float* reversal_values,
      int num_decay_terms,
      int* synapse_decay_values,
      float* neuron_wise_conductance_traces,
      float* d_neurons_current_injections,
      float * d_membrane_potentials_v,
      float timestep,
      size_t total_number_of_neurons);

    __global__ void conductance_move_spikes_towards_synapses_kernel(
      int* d_spikes_travelling_to_synapse,
      float current_time_in_seconds,
      int* circular_spikenum_buffer,
      int* spikeid_buffer,
      int bufferloc,
      int buffersize,
      int total_number_of_synapses,
      float* d_time_of_last_spike_to_reach_synapse,
      int* postsynaptic_neuron_indices,
      float* neuron_wise_conductance_trace,
      int* synapse_decay_id,
      int total_number_of_neurons,
      float* d_synaptic_efficacies_or_weights,
      float* d_biological_conductance_scaling_constants_lambda,
      float timestep);
  }
}
