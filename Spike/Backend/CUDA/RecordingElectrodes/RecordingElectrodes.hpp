#pragma once

#include "Spike/RecordingElectrodes/RecordingElectrodes.hpp"
#include "Spike/Backend/CUDA/CUDABackend.hpp"

#include "Spike/Neurons/SpikingNeurons.hpp"
#include "Spike/Backend/CUDA/Neurons/SpikingNeurons.hpp"

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
      void reset_state() override;

    protected:
      ::SpikingNeurons* neurons_frontend = nullptr;
      ::Backend::CUDA::SpikingNeurons* neurons_backend = nullptr;
    };
  }
}
