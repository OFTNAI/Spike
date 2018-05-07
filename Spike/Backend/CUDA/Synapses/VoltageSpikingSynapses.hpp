#pragma once

#include "Spike/Synapses/VoltageSpikingSynapses.hpp"
#include "SpikingSynapses.hpp"
#include "Spike/Backend/CUDA/CUDABackend.hpp"
#include <cuda.h>
#include <vector_types.h>
#include <curand.h>
#include <curand_kernel.h>

namespace Backend {
  namespace CUDA {
    struct voltage_spiking_synapses_data_struct: spiking_synapses_data_struct {
    };
    class VoltageSpikingSynapses : public virtual ::Backend::CUDA::SpikingSynapses,
                                       public virtual ::Backend::VoltageSpikingSynapses {
    public:

      ~VoltageSpikingSynapses() override;
      SPIKE_MAKE_BACKEND_CONSTRUCTOR(VoltageSpikingSynapses);
      using ::Backend::VoltageSpikingSynapses::frontend;

      conductance_spiking_synapses_data_struct* synaptic_data;
      conductance_spiking_synapses_data_struct* d_synaptic_data;

      void prepare() override;
      void reset_state() override;

      void allocate_device_pointers(); // Not virtual
      void copy_constants_and_initial_efficacies_to_device(); // Not virtual

      void state_update(::SpikingNeurons * neurons, ::SpikingNeurons * input_neurons, float current_time_in_seconds, float timestep) final;

    };
    __device__ float voltage_spiking_current_injection_kernel(
        spiking_synapses_data_struct* synaptic_data,
	      spiking_neurons_data_struct* neuron_data,
        float multiplication_to_volts,
        float current_membrane_voltage,
        float timestep,
        int timestep_grouping,
	      int idx,
	      int g);
  }
}
