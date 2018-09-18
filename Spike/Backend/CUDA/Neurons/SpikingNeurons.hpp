#pragma once

#include "Spike/Neurons/SpikingNeurons.hpp"
#include "Spike/Backend/CUDA/CUDABackend.hpp"
#include "Neurons.hpp"

#include <cuda.h>
#include <vector_types.h>

//#define BITLOC(current_time_in_seconds, timestep, offset, bufsizebytes) (((int)ceil(current_time_in_seconds / timestep) + offset) % bufsizebytes*8)
//(((int)(ceil(current_time_in_seconds / timestep)) + g) % (8*bufsizebytes))
#define BYTELOC(bitloc) (bitloc / 8)
#define SUBBITLOC(bitloc) (bitloc % 8)

namespace Backend {
  namespace CUDA {
    struct spiking_neurons_data_struct : neurons_data_struct {
        float* last_spike_time_of_each_neuron;
        float* membrane_potentials_v; 
        float* thresholds_for_action_potential_spikes;
        float* resting_potentials_v0;
        float* after_spike_reset_potentials_vreset;

        uint8_t* neuron_spike_time_bitbuffer;
        int* neuron_spike_time_bitbuffer_bytesize;
    };

    class SpikingNeurons : public virtual ::Backend::CUDA::Neurons,
                           public virtual ::Backend::SpikingNeurons {
    public:
      SpikingNeurons();
      ~SpikingNeurons() override;
      SPIKE_MAKE_BACKEND_CONSTRUCTOR(SpikingNeurons);
      using ::Backend::SpikingNeurons::frontend;

      void prepare() override;
      void reset_state() override;

      // Device Pointers
      float* last_spike_time_of_each_neuron = nullptr;
      float* membrane_potentials_v;
      float* thresholds_for_action_potential_spikes;
      float* resting_potentials_v0;
      float* after_spike_reset_potentials_vreset;

      // Keeping neuorn spike times
      int h_neuron_spike_time_bitbuffer_bytesize;
      int* neuron_spike_time_bitbuffer_bytesize = nullptr;
      uint8_t* neuron_spike_time_bitbuffer = nullptr;

      spiking_neurons_data_struct* neuron_data;
      spiking_neurons_data_struct* d_neuron_data;

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

  }
} // namespace Backend
