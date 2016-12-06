#pragma once

#include "Spike/RecordingElectrodes/RecordingElectrodes.hpp"
#include "Spike/Backend/CUDA/CUDABackend.hpp"
#include <cuda.h>
#include <vector_types.h>
#include <curand.h>
#include <curand_kernel.h>

namespace Backend {
  namespace CUDA {
    class RecordingElectrodes : public virtual ::Backend::RecordingElectrodes {
    public:
      using ::Backend::RecordingElectrodes::frontend;
      void prepare() override;
    protected:
      ::SpikingNeurons* neurons_frontend = nullptr;
      ::Backend::CUDA::SpikingNeurons* neurons_backend = nullptr;
    };
  }
}
