#pragma once

#include "Spike/Neurons/LIFSpikingNeurons.hpp"
#include "Spike/Backend/CUDA/CUDABackend.hpp"
#include "SpikingNeurons.hpp"

#include <cuda.h>
#include <vector_types.h>

namespace Backend {
  namespace CUDA {
    class LIFSpikingNeurons : public virtual ::Backend::CUDA::SpikingNeurons,
                              public virtual ::Backend::LIFSpikingNeurons {
    public:
      float * membrane_time_constants_tau_m = nullptr;
      float * membrane_resistances_R = nullptr;

      ~LIFSpikingNeurons();
      MAKE_BACKEND_CONSTRUCTOR(LIFSpikingNeurons);
      using ::Backend::LIFSpikingNeurons::frontend;

      // void check_for_neuron_spikes(float current_time_in_seconds, float timestep) override;
      // void reset_state() override;
      // void prepare() override;

      void allocate_device_pointers(int maximum_axonal_delay_in_timesteps, bool high_fidelity_spike_storage) override;
      void copy_constants_to_device() override;
      void update_membrane_potentials(float timestep, float current_time_in_seconds) override;

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
