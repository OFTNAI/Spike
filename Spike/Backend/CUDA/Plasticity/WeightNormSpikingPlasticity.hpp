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

      bool* neuron_in_set = nullptr;
      float* total_afferent_synapse_initial = nullptr;
      float* afferent_synapse_changes = nullptr;

      float* initial_weights = nullptr;
      float* weight_changes = nullptr;

      void prepare() override;
      void reset_state() override;

      void allocate_device_pointers();
      void weight_normalization();

    protected:
      ::Backend::CUDA::SpikingNeurons* neurons_backend = nullptr;
      ::Backend::CUDA::SpikingSynapses* synapses_backend = nullptr;

    };

  // CUDA Kernel for calculating synapse changes
  __global__ void weight_norm_change_detection


  }
}
