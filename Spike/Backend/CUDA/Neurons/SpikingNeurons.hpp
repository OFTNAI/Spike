#pragma once

#include "Spike/Neurons/SpikingNeurons.hpp"
#include "Spike/Backend/CUDA/CUDABackend.hpp"
#include "Neurons.hpp"

#include <cuda.h>
#include <vector_types.h>

namespace Backend {
  namespace CUDA {
    class SpikingNeurons : public virtual ::Backend::CUDA::Neurons,
                           public virtual ::Backend::SpikingNeurons {
    public:
      ~SpikingNeurons() override;

      void prepare() override;
      void reset_state() override;

      // Device Pointers
      float* last_spike_time_of_each_neuron;
      float* membrane_potentials_v;
      float* thresholds_for_action_potential_spikes;
      float* resting_potentials;

      /**  
       *  Exclusively for the allocation of device memory. This class requires allocation of d_current_injections only.
      */
      void allocate_device_pointers(); // Not virtual

      /**  
       *  Allows copying of static data related to neuron dynamics to the device.
       */
      void copy_constants_to_device(); // Not virtual

      void state_update(float current_time_in_seconds, float timestep) override;
    };

  } // namespace CUDA
} // namespace Backend
