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
      float * synaptic_conductances_g = nullptr;
      float * biological_conductance_scaling_constants_lambda = nullptr;
      float * reversal_potentials_Vhat = nullptr;
      float * decay_terms_tau_g = nullptr;
      int * num_active_synapses = nullptr;
      int * active_synapse_indices = nullptr;


      ~ConductanceSpikingSynapses() override;
      SPIKE_MAKE_BACKEND_CONSTRUCTOR(ConductanceSpikingSynapses);
      using ::Backend::ConductanceSpikingSynapses::frontend;
      
      void prepare() override;
      void reset_state() override;

      void allocate_device_pointers(); // Not virtual
      void copy_constants_and_initial_efficacies_to_device(); // Not virtual

      void calculate_postsynaptic_current_injection
      (::SpikingNeurons * neurons,
       float current_time_in_seconds,
       float timestep) final; // Overrides ::Backend::SpikingSynapses:: ...

      void update_synaptic_conductances(float timestep, float current_time_in_seconds) final;
      void interact_spikes_with_synapses(::SpikingNeurons * neurons, ::SpikingNeurons * input_neurons, float current_time_in_seconds, float timestep) final;

    };

    __global__ void get_active_synapses_kernel(int* d_per_neuron_efferent_synapse_count,
		int* d_per_neuron_efferent_synapse_total,
                int* d_per_neuron_efferent_synapse_indice,
                int* d_delays,
                int* d_spikes_travelling_to_synapse,
                float* d_last_spike_time_of_each_neuron,
                float * d_decay_terms_tau_g,
                float current_time_in_seconds,
                int* d_num_active_synapses,
                int* d_active_synapses,
                float timestep,
                size_t total_number_of_neurons);

    __global__ void conductance_calculate_postsynaptic_current_injection_kernel(int * d_presynaptic_neuron_indices,
              int* d_postsynaptic_neuron_indices,
              float* d_reversal_potentials_Vhat,
              float* d_neurons_current_injections,
              int* d_num_active_synapses,
              int* d_active_synapses,
              float * d_membrane_potentials_v,
              float * d_synaptic_conductances_g);

    __global__ void conductance_update_synaptic_conductances_kernel(float timestep,
                          float * d_synaptic_conductances_g,
                          float * d_synaptic_efficacies_or_weights,
                          float * d_time_of_last_spike_to_reach_synapse,
                          float * d_biological_conductance_scaling_constants_lambda,
                          int* d_num_active_synapses,
                          int* d_active_synapses,
                          float current_time_in_seconds,
                          float * d_decay_terms_tau_g);

    __global__ void conductance_move_spikes_towards_synapses_kernel(int* d_presynaptic_neuron_indices,
                int* d_delays,
                int* d_spikes_travelling_to_synapse,
                float* d_neurons_last_spike_time,
                float* d_input_neurons_last_spike_time,
                float currtime,
                int* d_num_active_synapses,
                int* d_active_synapses,
                float* d_time_of_last_spike_to_reach_synapse);

    __global__ void conductance_check_bitarray_for_presynaptic_neuron_spikes(int* d_presynaptic_neuron_indices,
                    int* d_delays,
                    unsigned char* d_bitarray_of_neuron_spikes,
                    unsigned char* d_input_neuruon_bitarray_of_neuron_spikes,
                    int bitarray_length,
                    int bitarray_maximum_axonal_delay_in_timesteps,
                    float current_time_in_seconds,
                    float timestep,
                    int* d_num_active_synapses,
                    int* d_active_synapses,
                    float* d_time_of_last_spike_to_reach_synapse);
  }
}
