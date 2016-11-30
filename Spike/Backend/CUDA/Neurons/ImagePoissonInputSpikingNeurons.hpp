#pragma once

#include "Spike/Neurons/PoissonInputSpikingNeurons.hpp"
#include "Spike/Backend/CUDA/CUDABackend.hpp"
#include "Neurons.hpp"

#include <cuda.h>
#include <vector_types.h>
#include <curand.h>
#include <curand_kernel.h>

namespace Backend {
  namespace CUDA {
    class ImagePoissonInputSpikingNeurons : public virtual ::Backend::CUDA::NeuronsCommon,
                                            public ::Backend::ImagePoissonInputSpikingNeurons {
    public:
      float * gabor_input_rates = nullptr;

      MAKE_BACKEND_CONSTRUCTOR(ImagePoissonInputSpikingNeurons);

      virtual void copy_rates_to_device();
      virtual void update_membrane_potentials(float timestep, float current_time_in_seconds);

      virtual void check_for_neuron_spikes(float current_time_in_seconds, float timestep);
      virtual void reset_state();
      virtual void prepare();
    };
  }
}
