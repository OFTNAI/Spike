#pragma once

#include "Spike/Synapses/SpikingSynapses.hpp"
#include "Spike/Backend/CUDA/CUDABackend.hpp"
#include "Spike/Helpers/CUDAErrorCheckHelpers.hpp"
#include <cuda.h>
#include <vector_types.h>
#include <curand.h>
#include <curand_kernel.h>

namespace Backend {
  namespace CUDA {
    class SpikingSynapses : public ::Backend::SpikingSynapses {
    public:
      // Device pointers
      int* delays = NULL;
      bool* stdp = NULL;
      int* spikes_travelling_to_synapse = NULL;
      float* time_of_last_spike_to_reach_synapse = NULL;

      virtual void prepare() {}
      virtual void reset_state() {}

      virtual void allocate_device_pointers();
      virtual void copy_constants_and_initial_efficacies_to_device();
      virtual void set_threads_per_block_and_blocks_per_grid(int threads);

      virtual void interact_spikes_with_synapses(SpikingNeurons * neurons, SpikingNeurons * input_neurons, float current_time_in_seconds, float timestep);
    };

    __global__ void move_spikes_towards_synapses_kernel(int* d_presynaptic_neuron_indices,
                                                        int* d_delays,
                                                        int* d_spikes_travelling_to_synapse,
                                                        float* d_neurons_last_spike_time,
                                                        float* d_input_neurons_last_spike_time,
                                                        float currtime,
                                                        size_t total_number_of_synapses,
                                                        float* d_time_of_last_spike_to_reach_synapse);

    __global__ void check_bitarray_for_presynaptic_neuron_spikes(int* d_presynaptic_neuron_indices,
                                                                 int* d_delays,
                                                                 unsigned char* d_bitarray_of_neuron_spikes,
                                                                 unsigned char* d_input_neuruon_bitarray_of_neuron_spikes,
                                                                 int bitarray_length,
                                                                 int bitarray_maximum_axonal_delay_in_timesteps,
                                                                 float current_time_in_seconds,
                                                                 float timestep,
                                                                 size_t total_number_of_synapses,
                                                                 float* d_time_of_last_spike_to_reach_synapse);
b  }
