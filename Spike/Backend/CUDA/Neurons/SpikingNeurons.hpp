#pragma once

#include "Spike/Neurons/SpikingNeurons.hpp"
#include "Spike/Backend/CUDA/CUDABackend.hpp"
#include "Neurons.hpp"

#include <cuda.h>
#include <vector_types.h>

namespace Backend {
  namespace CUDA {
    struct spiking_neurons_data_struct : neurons_data_struct {
        float* last_spike_time_of_each_neuron;
	float* membrane_potentials_v;
	float* thresholds_for_action_potential_spikes;
	float* resting_potentials;
	float* current_injections;
	float* total_current_conductance;
    };

    class SpikingNeurons : public virtual ::Backend::CUDA::Neurons,
                           public virtual ::Backend::SpikingNeurons {
    public:
      ~SpikingNeurons() override;

      void prepare() override;
      void reset_state() override;

      float* current_injections = nullptr;				/**< Device array for the storage of current to be injected into each neuron on each timestep. */
      float* total_current_conductance = nullptr;				/**< Device array for the total current conductance to be accounted for. */
      
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
      
      void reset_current_injections() ;
    };

  } // namespace CUDA
} // namespace Backend
