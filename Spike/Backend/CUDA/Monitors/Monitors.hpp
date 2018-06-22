#pragma once

#include "Spike/Monitors/Monitors.hpp"
#include "Spike/Backend/CUDA/CUDABackend.hpp"

#include "Spike/Neurons/SpikingNeurons.hpp"
#include "Spike/Backend/CUDA/Neurons/SpikingNeurons.hpp"

#include <cuda.h>
#include <vector_types.h>
#include <curand.h>
#include <curand_kernel.h>

namespace Backend {
  namespace CUDA {
    class Monitors : public virtual ::Backend::Monitors {
    public:
      using ::Backend::Monitors::frontend;

      void prepare() override;
      void reset_state() override;

    protected:
      ::SpikingNeurons* neurons_frontend = nullptr;
      ::Backend::CUDA::SpikingNeurons* neurons_backend = nullptr;
    };
  }
}
