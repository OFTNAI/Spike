#pragma once

#include "Spike/Neurons/PoissonInputSpikingNeurons.hpp"

#include "InputSpikingNeurons.hpp"

#include "Spike/Backend/CUDA/CUDABackend.hpp"
#include "Spike/Backend/CUDA/Helpers/RandomStateManager.hpp"

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
      ~PoissonInputSpikingNeurons() override;

      SPIKE_MAKE_BACKEND_CONSTRUCTOR(PoissonInputSpikingNeurons);
      using ::Backend::PoissonInputSpikingNeurons::frontend;

      void prepare() override;
      void reset_state() override;

      ::Backend::CUDA::RandomStateManager* random_state_manager_backend = nullptr;
      float * rates = nullptr;
      
      void allocate_device_pointers(); // Not virtual
      void copy_constants_to_device(); // Not virtual

      void update_membrane_potentials(float timestep, float current_time_in_secods) override;
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
