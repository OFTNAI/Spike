#pragma once

#include "Spike/SpikeAnalyser/SpikeAnalyser.hpp"
#include "Spike/Backend/CUDA/CUDABackend.hpp"
#include <cuda.h>
#include <vector_types.h>
#include <curand.h>
#include <curand_kernel.h>

namespace Backend {
  namespace CUDA {
    class SpikeAnalyserCommon : public virtual ::Backend::SpikeAnalyserCommon {
    public:
    };

    class SpikeAnalyser : public virtual ::Backend::CUDA::SpikeAnalyserCommon,
                          public ::Backend::SpikeAnalyser {
    public:
      MAKE_BACKEND_CONSTRUCTOR(SpikeAnalyser);

      virtual void store_spike_counts_for_stimulus_index(::SpikeAnalyser* front,
                                                         int stimulus_index);
    };
  }
}
