#pragma once

#include "Spike/Neurons/IzhikevichSpikingNeurons.hpp"
#include "Spike/Backend/CUDA/CUDABackend.hpp"
#include "Spike/Helpers/CUDAErrorCheckHelpers.hpp"
#include <cuda.h>
#include <vector_types.h>

namespace Backend {
  namespace CUDA {
    class IzhikevichSpikingNeurons : public ::Backend::IzhikevichSpikingNeurons {
    public:

      // Device Pointers
      float * param_a = NULL;
      float * param_b = NULL;
      float * param_d = NULL;
      float * states_u = NULL;

      virtual void allocate_device_pointers(int maximum_axonal_delay_in_timesteps, bool high_fidelity_spike_storage);
      virtual void copy_constants_to_device();
      virtual void check_for_neuron_spikes(float current_time_in_seconds, float timestep);
      virtual void update_membrane_potentials(float timestep, float current_time_in_seconds);
      virtual void reset();

    };

    __global__ void reset_states_u_after_spikes_kernel(float *d_states_u,
                                                       float * d_param_d,
                                                       float* d_last_spike_time_of_each_neuron,
                                                       float current_time_in_seconds,
                                                       size_t total_number_of_neurons);

    __global__ void izhikevich_update_membrane_potentials_kernel(float *d_membrane_potentials_v,
                                                                 float *d_states_u,
                                                                 float *d_param_a,
                                                                 float *d_param_b,
                                                                 float *d_current_injections,
                                                                 float timestep,
                                                                 size_t total_number_of_neurons);
  }
}
