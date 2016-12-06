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
      ~SpikingNeurons();

      // Device Pointers
      float* last_spike_time_of_each_neuron;
      float* membrane_potentials_v;
      float* thresholds_for_action_potential_spikes;
      float* resting_potentials;
      unsigned char * bitarray_of_neuron_spikes;

      /**  
       *  Exclusively for the allocation of device memory. This class requires allocation of d_current_injections only.
       \param maximum_axonal_delay_in_timesteps The length (in timesteps) of the largest axonal delay in the simulation. Unused in this class.
       \param high_fidelity_spike_storage A flag determining whether a bit mask based method is used to store spike times of neurons (ensure no spike transmission failure).
      */
      virtual void allocate_device_pointers(int maximum_axonal_delay_in_timesteps,  bool high_fidelity_spike_flag);

      virtual void check_for_neuron_spikes(float current_time_in_seconds, float timestep);

      virtual void reset_state();
      virtual void prepare();
      
      /**  
       *  Unused in this class. Allows copying of static data related to neuron dynamics to the device.
       */
      virtual void copy_constants_to_device();

    private:
      ADD_FRONTEND_GETTER(SpikingNeurons);
    };

    __global__ void check_for_neuron_spikes_kernel(float *d_membrane_potentials_v,
                                                   float *d_thresholds_for_action_potential_spikes,
                                                   float *d_resting_potentials,
                                                   float* d_last_spike_time_of_each_neuron,
                                                   unsigned char* d_bitarray_of_neuron_spikes,
                                                   int bitarray_length,
                                                   int bitarray_maximum_axonal_delay_in_timesteps,
                                                   float current_time_in_seconds,
                                                   float timestep,
                                                   size_t total_number_of_neurons,
                                                   bool high_fidelity_spike_flag);
  } // namespace CUDA
} // namespace Backend
