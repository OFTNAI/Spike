#pragma once

#include "Spike/Neurons/GeneratorInputSpikingNeurons.hpp"
#include "InputSpikingNeurons.hpp"
#include "Spike/Backend/CUDA/CUDABackend.hpp"

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

      void allocate_device_pointers(); // Not virtual

      void check_for_neuron_spikes(float current_time_in_seconds, float timestep) override;
      void update_membrane_potentials(float timestep, float current_time_in_seconds) override;
    };

    __global__ void check_for_generator_spikes_kernel(int *d_neuron_ids_for_stimulus,
                                                      float *d_spike_times_for_stimulus,
                                                      float* d_last_spike_time_of_each_neuron,
                                                      unsigned char* d_bitarray_of_neuron_spikes,
                                                      int bitarray_length,
                                                      int bitarray_maximum_axonal_delay_in_timesteps,
                                                      float current_time_in_seconds,
                                                      float timestep,
                                                      size_t number_of_spikes_in_stimulus,
                                                      bool high_fidelity_spike_flag);

  }
}
