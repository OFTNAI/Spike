#pragma once

#include "Spike/Plasticity/WeightNormSpikingPlasticity.hpp"
#include "Plasticity.hpp"

#include "Spike/Backend/CUDA/Neurons/SpikingNeurons.hpp"
#include "Spike/Backend/CUDA/Synapses/SpikingSynapses.hpp"

#include "Spike/Backend/CUDA/CUDABackend.hpp"
#include <cuda.h>
#include <vector_types.h>
#include <curand.h>
#include <curand_kernel.h>

namespace Backend {
  namespace CUDA {
    class WeightNormSpikingPlasticity : public virtual ::Backend::CUDA::STDPPlasticity,
                           public virtual ::Backend::WeightNormSpikingPlasticity {
    public:
      ~WeightNormSpikingPlasticity() override;
      SPIKE_MAKE_BACKEND_CONSTRUCTOR(WeightNormSpikingPlasticity);
      using ::Backend::WeightNormSpikingPlasticity::frontend;

      int total_number_of_plastic_synapses = 0;
      int* plastic_synapse_indices = nullptr;

      bool* neuron_in_plasticity_set = nullptr;
      float* total_afferent_synapse_initial = nullptr;
      float* afferent_synapse_changes = nullptr;

      float* initial_weights = nullptr;
      float* weight_changes = nullptr;

      void prepare() override;
      void reset_state() override;

      void allocate_device_pointers();
      virtual void weight_normalization() ovverride;

    protected:
      ::Backend::CUDA::Synapses* synapses_backend = nullptr;

    };

  // CUDA Kernel for calculating synapse changes
  __global__ void weight_change_calculations
	(float* current_weight,
	 float* initial_weights,
         float* weight_changes,
	 int* d_plastic_synapse_indices,
	 size_t total_number_of_plastic_synapses);


  // CUDA Kernel for weight change summation
  


  }
}
