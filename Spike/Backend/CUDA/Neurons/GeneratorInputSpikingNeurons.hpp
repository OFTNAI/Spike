#pragma once

#include "Spike/Neurons/GeneratorInputSpikingNeurons.hpp"
#include "InputSpikingNeurons.hpp"
#include "Spike/Backend/CUDA/CUDABackend.hpp"
#include "Spike/Backend/CUDA/Synapses/SpikingSynapses.hpp"

#include <cuda.h>
#include <vector_types.h>
#include <curand.h>
#include <curand_kernel.h>

namespace Backend {
  namespace CUDA {
    class GeneratorInputSpikingNeurons : public virtual ::Backend::CUDA::InputSpikingNeurons,
                                         public virtual ::Backend::GeneratorInputSpikingNeurons {
    public:
      ~GeneratorInputSpikingNeurons() override;

      SPIKE_MAKE_BACKEND_CONSTRUCTOR(GeneratorInputSpikingNeurons);
      using ::Backend::GeneratorInputSpikingNeurons::frontend;
      
      void prepare() override;
      void reset_state() override;

      // Device Pointers
      int* neuron_ids_for_stimulus = nullptr;
      float* spike_times_for_stimulus = nullptr;
      int num_spikes_in_current_stimulus = 0;

      void allocate_device_pointers(); // Not virtual

      void state_update(float current_time_in_seconds, float timestep) override;
    };

    __global__ void check_for_generator_spikes_kernel(
        synaptic_activation_kernel syn_activation_kernel,
        spiking_synapses_data_struct* synaptic_data,
        spiking_neurons_data_struct* in_neuron_data,
        int *d_neuron_ids_for_stimulus,
        float *d_spike_times_for_stimulus,
        float* d_last_spike_time_of_each_neuron,
        float current_time_in_seconds,
        float stimulus_onset_adjustment,
        float timestep,
        int timstep_index,
        int timestep_grouping,
        size_t number_of_spikes_in_stimulus);

  }
}
