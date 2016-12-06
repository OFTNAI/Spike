#pragma once

#include "Spike/Neurons/ImagePoissonInputSpikingNeurons.hpp"
#include "Spike/Backend/CUDA/CUDABackend.hpp"
#include "PoissonInputSpikingNeurons.hpp"

#include <cuda.h>
#include <vector_types.h>
#include <curand.h>
#include <curand_kernel.h>

namespace Backend {
  namespace CUDA {
    class ImagePoissonInputSpikingNeurons : public virtual ::Backend::CUDA::PoissonInputSpikingNeurons,
                                            public virtual ::Backend::ImagePoissonInputSpikingNeurons {
    public:
      ~ImagePoissonInputSpikingNeurons();
      
      float * gabor_input_rates = nullptr;

      MAKE_BACKEND_CONSTRUCTOR(ImagePoissonInputSpikingNeurons);
      using ::Backend::ImagePoissonInputSpikingNeurons::frontend;

      virtual void copy_rates_to_device();
      virtual void update_membrane_potentials(float timestep, float current_time_in_seconds);

      virtual void check_for_neuron_spikes(float current_time_in_seconds, float timestep);
      virtual void reset_state();
      virtual void prepare();
    };
  }
}
