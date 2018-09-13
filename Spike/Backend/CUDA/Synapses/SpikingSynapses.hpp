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

    enum SYNAPSE_TYPE
    {
      EMPTY,
      CONDUCTANCE,
      CURRENT,
      VOLTAGE
    };
    struct neuron_inputs_struct {
      float* circular_input_buffer = nullptr;
      int* bufferloc = nullptr;
      int input_buffersize = 0;
      int temporal_buffersize = 0;
    };
    struct spiking_synapses_data_struct: synapses_data_struct {
      neuron_inputs_struct neuron_inputs;
      int synapse_type = EMPTY;
      int num_syn_labels = 0;
      int* num_activated_neurons = nullptr;
      int* num_active_synapses = nullptr;
      int* active_synapse_counts = nullptr;
      int* presynaptic_neuron_indices = nullptr;
      int* group_indices = nullptr;

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
    typedef void (*synaptic_activation_kernel)(
      spiking_synapses_data_struct* synaptic_data,
      spiking_neurons_data_struct* neuron_data,
      int timestep_group_index,
      int preneuron_idx,
      bool is_input);
    class SpikingSynapses : public virtual ::Backend::CUDA::Synapses,
                            public virtual ::Backend::SpikingSynapses {
    public:
      SPIKE_MAKE_BACKEND_CONSTRUCTOR(SpikingSynapses);
      
      // Variables used to determine active/inactive synapses
      int buffersize = 0;
      int* group_indices = nullptr;
      int* num_active_synapses = nullptr;
      int* num_activated_neurons = nullptr;
      int* active_synapse_counts = nullptr;
      int* presynaptic_neuron_indices = nullptr;
      int h_num_active_synapses = 0;
      // Device pointers
      int* delays = nullptr;
      int* spikes_travelling_to_synapse = nullptr;
      float* time_of_last_spike_to_reach_synapse = nullptr;

      int* d_syn_labels = nullptr;
      neuron_inputs_struct neuron_inputs;

      SpikingSynapses();
      ~SpikingSynapses() override;
      using ::Backend::SpikingSynapses::frontend;

      spiking_synapses_data_struct* synaptic_data;
      spiking_synapses_data_struct* d_synaptic_data;
      injection_kernel host_injection_kernel;
      synaptic_activation_kernel host_syn_activation_kernel;

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

    __device__ void get_active_synapses(
      spiking_synapses_data_struct* synaptic_data,
      spiking_neurons_data_struct* neuron_data,
      int timestep_group_index,
      int preneuron_idx,
      bool is_input);

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
        int bufferloc,
        int buffersize,
        neuron_inputs_struct neuron_inputs,
        int* postsynaptic_neuron_indices,
        float* synaptic_efficacies_or_weights,
        float* weight_scaling_constants,
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
  }
}
