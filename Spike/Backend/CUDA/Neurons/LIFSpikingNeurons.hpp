#pragma once

#include "Spike/Neurons/LIFSpikingNeurons.hpp"
#include "Spike/Backend/CUDA/CUDABackend.hpp"
#include <cuda.h>
#include <vector_types.h>

namespace Backend {
  namespace CUDA {
    class LIFSpikingNeurons : public ::Backend::LIFSpikingNeurons {
    public:
      float * membrane_time_constants_tau_m = NULL;
      float * membrane_resistances_R = NULL;

      virtual void allocate_device_pointers(int maximum_axonal_delay_in_timesteps, bool high_fidelity_spike_storage);
      virtual void copy_constants_to_device();
      virtual void update_membrane_potentials(float timestep, float current_time_in_seconds);

    };

    __global__ void lif_update_membrane_potentials(float *d_membrane_potentials_v,
                                                   float * d_last_spike_time_of_each_neuron,
                                                   float * d_membrane_resistances_R,
                                                   float * d_membrane_time_constants_tau_m,
                                                   float * d_resting_potentials,
                                                   float* d_current_injections,
                                                   float timestep,
                                                   float current_time_in_seconds,
                                                   float refactory_period_in_seconds,
                                                   size_t total_number_of_neurons);
  }
}
