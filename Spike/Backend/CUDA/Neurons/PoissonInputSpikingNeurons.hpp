#pragma once

#include "Spike/Neurons/PoissonInputSpikingNeurons.hpp"
#include "Spike/Backend/CUDA/CUDABackend.hpp"

#include <cuda.h>
#include <vector_types.h>
#include <curand.h>
#include <curand_kernel.h>

namespace Backend {
  namespace CUDA {
    class PoissonInputSpikingNeurons : public ::Backend::PoissonInputSpikingNeurons {
    public:
      float * d_rates = NULL;
      
      virtual void allocate_device_pointers(int maximum_axonal_delay_in_timesteps, bool high_fidelity_spike_storage);
      virtual void copy_constants_to_device();
      virtual void update_membrane_potentials(float timestep, float current_time_in_seconds);
      // virtual void setup_random_states_on_device();
    };

    __global__ void poisson_update_membrane_potentials_kernel(curandState_t* d_states,
                                                              float *d_rates,
                                                              float *d_membrane_potentials_v,
                                                              float timestep,
                                                              float * d_thresholds_for_action_potential_spikes,
                                                              size_t total_number_of_input_neurons,
                                                              int current_stimulus_index);
  }
}
