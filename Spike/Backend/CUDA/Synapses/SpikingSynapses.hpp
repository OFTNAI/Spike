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
    class SpikingSynapses : public virtual ::Backend::CUDA::Synapses,
                            public virtual ::Backend::SpikingSynapses {
    public:
      // Device pointers
      int* delays = nullptr;
      bool* stdp = nullptr;
      int* spikes_travelling_to_synapse = nullptr;
      float* time_of_last_spike_to_reach_synapse = nullptr;

      ~SpikingSynapses() override;
      using ::Backend::SpikingSynapses::frontend;

      void prepare() override;
      void reset_state() override;

      void allocate_device_pointers(); // Not virtual
      void copy_constants_and_initial_efficacies_to_device(); // Not virtual

      void copy_weights_to_host() override;

      void interact_spikes_with_synapses(::SpikingNeurons * neurons, ::SpikingNeurons * input_neurons, float current_time_in_seconds, float timestep) final;
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
  }
}
