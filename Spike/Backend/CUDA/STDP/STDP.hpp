#pragma once

#include "Spike/STDP/STDP.hpp"
#include "Spike/Backend/CUDA/CUDABackend.hpp"
#include <cuda.h>
#include <vector_types.h>
#include <curand.h>
#include <curand_kernel.h>

namespace Backend {
  namespace CUDA {
    class STDP : public virtual ::Backend::STDP {
    public:
      void prepare() override;
      using ::Backend::STDP::frontend;
    protected:
      ::Backend::CUDA::SpikingNeurons* neurons_backend = nullptr;
      ::Backend::CUDA::SpikingSynapses* synapses_backend = nullptr;
    };
  }
}
