#pragma once

#include "Spike/ActivityMonitor/ActivityMonitor.hpp"
#include "Spike/Backend/CUDA/CUDABackend.hpp"

#include "Spike/Neurons/SpikingNeurons.hpp"
#include "Spike/Backend/CUDA/Neurons/SpikingNeurons.hpp"

#include <cuda.h>
#include <vector_types.h>
#include <curand.h>
#include <curand_kernel.h>

namespace Backend {
  namespace CUDA {
    class ActivityMonitor : public virtual ::Backend::ActivityMonitor {
    public:
      using ::Backend::ActivityMonitor::frontend;

      void prepare() override;
      void reset_state() override;

    protected:
      ::SpikingNeurons* neurons_frontend = nullptr;
      ::Backend::CUDA::SpikingNeurons* neurons_backend = nullptr;
    };
  }
}
