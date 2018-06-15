#pragma once

#include "Synapses.hpp"

#include "Spike/Synapses/SpikingSynapses.hpp"
#include "Spike/Backend/CUDA/CUDABackend.hpp"
#include "Spike/Backend/CUDA/Neurons/SpikingNeurons.hpp"

#include <cuda.h>
#include <vector_types.h>
#include <curand.h>
#include <curand_kernel.h>

namespace Backend {
  namespace CUDA {
    struct neuron_inputs_struct {
      float* circular_input_buffer = nullptr;
      int* bufferloc = nullptr;
      int input_buffersize = 0;
      int temporal_buffersize = 0;
    };
    struct spiking_synapses_data_struct: synapses_data_struct {
      float* neuron_wise_input_update;
      neuron_inputs_struct neuron_inputs;
      int num_syn_labels;
    };
    typedef float (*injection_kernel)(
        spiking_synapses_data_struct* synaptic_data,
        spiking_neurons_data_struct* neuron_data,
        float multiplication_to_volts,
        float membrane_voltage,
        float current_time_in_seconds,
        float timestep,
        int timestep_grouping,
        int idx,
        int g);
    class SpikingSynapses : public virtual ::Backend::CUDA::Synapses,
                            public virtual ::Backend::SpikingSynapses {
    public:
      
      // Variables used to determine active/inactive synapses
      int* circular_spikenum_buffer = nullptr;
      int* spikeid_buffer = nullptr;
      int buffersize = 0;
      int* group_indices = nullptr;
      int* num_active_synapses = nullptr;
      int* num_activated_neurons = nullptr;
      int* activated_neuron_afferents = nullptr;
      int* activated_neuron_afferents_count = nullptr;
      int* active_synapse_counts = nullptr;
      int* presynaptic_neuron_indices = nullptr;
      int h_num_active_synapses = 0;
      // Device pointers
      int* delays = nullptr;
      int* spikes_travelling_to_synapse = nullptr;
      float* time_of_last_spike_to_reach_synapse = nullptr;
      float * biological_conductance_scaling_constants_lambda = nullptr;

      int neuron_wise_input_length = 0;
      float* neuron_wise_input_update = nullptr;
      float* h_neuron_wise_input_update = nullptr;
      int* d_syn_labels = nullptr;
      neuron_inputs_struct neuron_inputs;

      ~SpikingSynapses() override;
      using ::Backend::SpikingSynapses::frontend;

      spiking_synapses_data_struct* synaptic_data;
      spiking_synapses_data_struct* d_synaptic_data;
      injection_kernel host_injection_kernel;

      void prepare() override;
      void reset_state() override;

      void allocate_device_pointers(); // Not virtual
      void copy_constants_and_initial_efficacies_to_device(); // Not virtual

      void copy_weights_to_host() override;

      void state_update(::SpikingNeurons * neurons, ::SpikingNeurons * input_neurons, float current_time_in_seconds, float timestep) override;

    };

      __device__ float spiking_current_injection_kernel(
  spiking_synapses_data_struct* synaptic_data,
  spiking_neurons_data_struct* neuron_data,
  float multiplication_to_volts,
  float current_membrane_voltage,
  float current_time_in_seconds,
  float timestep,
  int timestep_grouping,
  int idx,
  int g);


    __global__ void get_active_synapses_kernel(
      int* d_per_neuron_efferent_synapse_count,
      int* d_per_input_neuron_efferent_synapse_count,
      float* d_last_spike_time_of_each_neuron,
      float* d_last_spike_time_of_each_input_neuron,
      float current_time_in_seconds,
      float timestep,
      int num_input_neurons,
      int* group_indices,
      int* num_active_synapses,
      int* num_activated_neurons,
      int* active_synapse_counts,
      int* presynaptic_neuron_indices,
      size_t total_number_of_neurons);
    
    __global__ void activate_synapses(
        int* d_per_neuron_efferent_synapse_total,
        int* d_per_neuron_efferent_synapse_indices,
        int* d_per_input_neuron_efferent_synapse_total,
        int* d_per_input_neuron_efferent_synapse_indices,
        int* circular_spikenum_buffer,
        int* spikeid_buffer,
        int bufferloc,
        int buffersize,
        neuron_inputs_struct neuron_inputs,
        int neuron_input_length,
        int* postsynaptic_neuron_indices,
        float* synaptic_efficacies_or_weights,
        float* biological_conductance_scaling_constants_lambda,
        float* last_spike_to_reach_synapse,
        int* d_delays,
        int * d_syn_labels,
        float timestep,
        float current_time_in_seconds,
        int total_number_of_synapses,
        int total_number_of_neurons,
        int* group_indices,
        int timestep_grouping,
        int* presynaptic_neuron_indices,
        int* active_synapse_counts,
        int* num_active_synapses);

    __global__ void move_spikes_towards_synapses_kernel(
      float current_time_in_seconds,
      int* circular_spikenum_buffer,
      int* spikeid_buffer,
      int bufferloc,
      int buffersize,
      int total_number_of_synapses,
      float* d_time_of_last_spike_to_reach_synapse,
      int* postsynaptic_neuron_indices,
      float* neuron_wise_input_update,
      int* synapse_decay_id,
      int num_syn_labels,
      int total_number_of_neurons,
      float* d_synaptic_efficacies_or_weights,
      float* d_biological_conductance_scaling_constants_lambda,
      float timestep,
      int timestep_grouping);

  }
}
