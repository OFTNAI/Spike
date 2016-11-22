#pragma once

#include "Spike/RecordingElectrodes/RecordingElectrodes.hpp"
#include "Spike/Backend/CUDA/CUDABackend.hpp"
#include <cuda.h>
#include <vector_types.h>
#include <curand.h>
#include <curand_kernel.h>

namespace Backend {
  namespace CUDA {
    class RecordingElectrodesCommon : public virtual ::Backend::RecordingElectrodesCommon {
    public:
    };

    class RecordingElectrodes : public virtual ::Backend::CUDA::RecordingElectrodesCommon,
                                public ::Backend::RecordingElectrodes {
    public:
    };
  }
}
