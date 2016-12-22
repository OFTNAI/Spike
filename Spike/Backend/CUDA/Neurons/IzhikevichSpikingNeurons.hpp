#pragma once

#include "Spike/Neurons/IzhikevichSpikingNeurons.hpp"
#include "SpikingNeurons.hpp"
#include "Spike/Backend/CUDA/CUDABackend.hpp"

#include <cuda.h>
#include <vector_types.h>

namespace Backend {
  namespace CUDA {
    class IzhikevichSpikingNeurons : public virtual ::Backend::CUDA::SpikingNeurons,
                                     public virtual ::Backend::IzhikevichSpikingNeurons {
    public:
      ~IzhikevichSpikingNeurons() override;

      SPIKE_MAKE_BACKEND_CONSTRUCTOR(IzhikevichSpikingNeurons);
      using ::Backend::IzhikevichSpikingNeurons::frontend;

      void prepare() override;
      void reset_state() override;

      void push_data_front() override;
      void pull_data_back() override;

      // Device Pointers
      float * param_a = nullptr;
      float * param_b = nullptr;
      float * param_d = nullptr;
      float * states_u = nullptr;

      void allocate_device_pointers(); // Not virtual
      void copy_constants_to_device(); // Not virtual

      void check_for_neuron_spikes(float current_time_in_seconds, float timestep) override;
      void update_membrane_potentials(float timestep, float current_time_in_seconds) override;
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
