#pragma once

#include "SpikingSynapses.hpp"

#include "Spike/Synapses/CurrentSpikingSynapses.hpp"
#include "Spike/Backend/CUDA/CUDABackend.hpp"
#include "Spike/Backend/CUDA/Neurons/SpikingNeurons.hpp"

#include <cuda.h>
#include <vector_types.h>
#include <curand.h>
#include <curand_kernel.h>

namespace Backend {
  namespace CUDA {
    struct current_spiking_synapses_data_struct: spiking_synapses_data_struct {
	    float* neuron_wise_current_trace;
      float* decay_terms_tau;
    };
    class CurrentSpikingSynapses : public virtual ::Backend::CUDA::SpikingSynapses,
                                   public virtual ::Backend::CurrentSpikingSynapses {
    public:
      ~CurrentSpikingSynapses();
      SPIKE_MAKE_BACKEND_CONSTRUCTOR(CurrentSpikingSynapses);
      using ::Backend::CurrentSpikingSynapses::frontend;
      
      int current_array_length = 0;
      float* neuron_wise_current_trace = nullptr;
      float* h_neuron_wise_current_trace = nullptr;
      float* d_decay_terms_tau = nullptr;
      
      void prepare() override;
      void reset_state() override;

      void allocate_device_pointers(); // Not virtual
      void copy_constants_and_initial_efficacies_to_device(); // Not virtual

      void state_update
      (::SpikingNeurons * neurons,
       ::SpikingNeurons* input_neurons,
       float current_time_in_seconds,
       float timestep) final; // Overrides ::Backend::SpikingSynapses:: ...
      
    };
    __device__ float current_spiking_current_injection_kernel(
        spiking_synapses_data_struct* synaptic_data,
        spiking_neurons_data_struct* neuron_data,
        float multiplication_to_volts,
        float current_membrane_voltage,
        float current_time_in_seconds,
        float timestep,
        int idx,
        int g);
  }
}

