#pragma once

#include "Spike/Neurons/LIFSpikingNeurons.hpp"
#include "Spike/Neurons/SpikingNeurons.hpp"
#include "Spike/Backend/CUDA/CUDABackend.hpp"
#include "Spike/Backend/CUDA/Neurons/SpikingNeurons.hpp"
#include "Spike/Backend/CUDA/Synapses/ConductanceSpikingSynapses.hpp"

#include <cuda.h>
#include <vector_types.h>

namespace Backend {
  namespace CUDA {
    struct lif_spiking_neurons_data_struct: spiking_neurons_data_struct {
	float* membrane_time_constants_tau_m;
	float* membrane_resistances_R;

    };

    class LIFSpikingNeurons : public virtual ::Backend::CUDA::SpikingNeurons,
                              public virtual ::Backend::LIFSpikingNeurons {
    public:
      float * membrane_time_constants_tau_m = nullptr;
      float * membrane_resistances_R = nullptr;

      ~LIFSpikingNeurons() override;
      SPIKE_MAKE_BACKEND_CONSTRUCTOR(LIFSpikingNeurons);
      using ::Backend::LIFSpikingNeurons::frontend;

      lif_spiking_neurons_data_struct* neuron_data;
      lif_spiking_neurons_data_struct* d_neuron_data;

      void prepare() override;
      void reset_state() override;

      void copy_constants_to_device(); // Not virtual
      void allocate_device_pointers(); // Not virtual

      void state_update(float current_time_in_seconds, float timestep) override;
    };

    __global__ void lif_update_membrane_potentials(
        injection_kernel current_injection_kernel,
        spiking_synapses_data_struct* synaptic_data,
	      spiking_neurons_data_struct* neuron_data,
        float background_current,
        float timestep,
				int timestep_grouping,
        float current_time_in_seconds,
        float refactory_period_in_seconds,
        size_t total_number_of_neurons);
  }
}
