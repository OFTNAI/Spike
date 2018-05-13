#pragma once

#include "Spike/Plasticity/STDPPlasticity.hpp"
#include "Spike/Backend/CUDA/Plasticity/Plasticity.hpp"
#include "Spike/Backend/CUDA/CUDABackend.hpp"
#include "Spike/Backend/CUDA/Neurons/SpikingNeurons.hpp"
#include "Spike/Backend/CUDA/Synapses/SpikingSynapses.hpp"

#include <cuda.h>
#include <vector_types.h>
#include <curand.h>
#include <curand_kernel.h>

namespace Backend {
  namespace CUDA {
    struct device_circular_spike_buffer_struct{
      int* time_buffer;
      int* id_buffer;
      int buffer_size;
    };
    class STDPPlasticity : public virtual ::Backend::CUDA::Plasticity,
        public virtual ::Backend::STDPPlasticity {
    protected:
      ::Backend::CUDA::SpikingNeurons* neurons_backend = nullptr;
      ::Backend::CUDA::SpikingSynapses* synapses_backend = nullptr;
    public:
      ~STDPPlasticity() override;
      using ::Backend::STDPPlasticity::frontend;
      int* plastic_synapse_indices = nullptr;
      int total_number_of_plastic_synapses;
      
      int* post_neuron_set = nullptr;
      int* pre_neuron_set = nullptr;
      int* pre_neuron_efferent_totals = nullptr;
      vector<int> h_pre_neuron_efferent_totals;
      int* pre_neuron_efferent_counts = nullptr;
      int* pre_neuron_efferent_ids = nullptr;
      vector<int> h_pre_neuron_efferent_ids;
      // Spike Buffer data location
      device_circular_spike_buffer_struct spike_buffer;
      int* num_activated_neurons = nullptr;
      int* activated_neuron_ids = nullptr;
      int* activated_neuron_groupindices = nullptr;
      int* num_active_synapses = nullptr;
      int h_num_active_synapses;
      
      void update_active_plastic_elements(float current_time_in_seconds, float timestep);

      void prepare() override;
      void reset_state() override;
      void allocate_device_pointers();

    };

    __global__ void get_active_preneurons_kernel(
        int* d_pre_neuron_set,
        int* d_pre_neuron_efferent_counts,
        float* d_last_spike_time_of_each_neuron,
        float* d_last_spike_time_of_each_input_neuron,
        int* d_num_activated_neurons,
        int* d_activated_neuron_ids,
        int* d_activated_neuron_groupindices,
        int* d_num_active_synapses,
        float current_time_in_seconds,
        float timestep,
        size_t total_number_of_pre_neurons);

    __global__ void synapses_to_buffer_kernel(
        device_circular_spike_buffer_struct spike_buffer,
        int bufferloc,
        int* d_delays,
        int total_number_of_plastic_synapses,
        int timestep_grouping,
        int* d_pre_neuron_efferent_totals,
        int* d_pre_neuron_efferent_counts,
        int* d_pre_neuron_efferent_ids,
        int* d_activated_neuron_ids,
        int* d_activated_neuron_groupindices,
        int* d_num_active_synapses);
        
  }
}
