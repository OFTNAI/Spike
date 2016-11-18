#pragma once

#include "Spike/Neurons/PoissonInputSpikingNeurons.hpp"
#include "Spike/Backend/CUDA/CUDABackend.hpp"
#include "Spike/Helpers/CUDAErrorCheckHelpers.hpp"

#include <cuda.h>
#include <vector_types.h>
#include <curand.h>
#include <curand_kernel.h>

namespace Backend {
  namespace CUDA {
    class ImagePoissonInputSpikingNeurons : public ::Backend::ImagePoissonInputSpikingNeurons {
    public:
      float * gabor_input_rates = NULL;
      
      virtual void copy_rates_to_device();
      virtual void update_membrane_potentials(float timestep, float current_time_in_seconds);
    };
  }
}
