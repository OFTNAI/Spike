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
      int input_buffersize = 0;
      int temporal_buffersize = 0;
    };
    struct spiking_synapses_data_struct: synapses_data_struct {
      neuron_inputs_struct neuron_inputs;
      int synapse_type = EMPTY;
      int num_syn_labels = 0;
      int* syn_labels = nullptr;
      int* num_activated_neurons = nullptr;
      int* num_active_synapses = nullptr;
      int* active_synapse_counts = nullptr;
      int* active_synapse_starts = nullptr;
      int* active_presynaptic_neuron_indices = nullptr;
      int* group_indices = nullptr;

      int* postsynaptic_neuron_indices = nullptr;
      int* delays = nullptr;
      float* synaptic_efficacies_or_weights = nullptr;
      float* weight_scaling_constants = nullptr;
    };
    typedef float (*injection_kernel)(
        spiking_synapses_data_struct* synaptic_data,
        spiking_neurons_data_struct* neuron_data,
        float multiplication_to_volts,
        float membrane_voltage,
        float current_time_in_seconds,
        float timestep,
        int idx,
        int g);
    typedef void (*synaptic_activation_kernel)(
        spiking_synapses_data_struct* synaptic_data,
        spiking_neurons_data_struct* neuron_data,
        int timestep_group_index,
        int preneuron_idx,
        int timestep_index,
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
      int* active_synapse_starts = nullptr;
      int* active_presynaptic_neuron_indices = nullptr;
      int h_num_active_synapses = 0;
      // Device pointers
      int* delays = nullptr;

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
      int idx,
      int g);

    __device__ void get_active_synapses(
      spiking_synapses_data_struct* synaptic_data,
      spiking_neurons_data_struct* neuron_data,
      int timestep_group_index,
      int preneuron_idx,
      int timestep_index,
      bool is_input);
    
    __global__ void activate_synapses(
        spiking_synapses_data_struct* synaptic_data,
        spiking_neurons_data_struct* neurons_data,
        spiking_neurons_data_struct* in_neurons_data,
        int bufferloc,
        float timestep,
        float current_time_in_seconds,
        int timestep_index,
        int timestep_grouping);
  }
}
