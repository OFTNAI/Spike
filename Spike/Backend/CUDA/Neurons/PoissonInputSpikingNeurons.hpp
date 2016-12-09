#pragma once

#include "Spike/Neurons/PoissonInputSpikingNeurons.hpp"
#include "Spike/Backend/CUDA/CUDABackend.hpp"
#include "InputSpikingNeurons.hpp"

#include <cuda.h>
#include <vector_types.h>
#include <curand.h>
#include <curand_kernel.h>

namespace Backend {
  namespace CUDA {
    class PoissonInputSpikingNeurons : public virtual ::Backend::CUDA::InputSpikingNeurons,
                                       public virtual ::Backend::PoissonInputSpikingNeurons {
    public:
      PoissonInputSpikingNeurons() = default;
      ~PoissonInputSpikingNeurons();

      MAKE_BACKEND_CONSTRUCTOR(PoissonInputSpikingNeurons);
      using ::Backend::PoissonInputSpikingNeurons::frontend;

      ::Backend::CUDA::RandomStateManager* random_state_manager_backend = nullptr;
      float * rates = nullptr;
      
      void allocate_device_pointers(int maximum_axonal_delay_in_timesteps, bool high_fidelity_spike_storage) override;
      void copy_constants_to_device() override;
      void update_membrane_potentials(float timestep, float current_time_in_seconds) override;
      // virtual void setup_random_states_on_device();

      void reset_state() override;
      void prepare() override;
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
