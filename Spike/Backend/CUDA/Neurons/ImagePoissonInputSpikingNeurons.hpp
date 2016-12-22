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
      ~ImagePoissonInputSpikingNeurons() override;
      
      SPIKE_MAKE_BACKEND_CONSTRUCTOR(ImagePoissonInputSpikingNeurons);
      using ::Backend::ImagePoissonInputSpikingNeurons::frontend;

      void prepare() override;
      void reset_state() override;

      void push_data_front() override;
      void pull_data_back() override;

      float * gabor_input_rates = nullptr;

      void allocate_device_pointers(); // Not virtual
      void copy_rates_to_device(); // Not virtual

      void update_membrane_potentials(float timestep, float current_time_in_seconds) override;
    };
  }
}
