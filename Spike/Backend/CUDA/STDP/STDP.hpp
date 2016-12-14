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
      using ::Backend::STDP::frontend;

      void prepare() override;
      void reset_state() override;

      void push_data_front() override;
      void pull_data_back() override;
protected:
      ::Backend::CUDA::SpikingNeurons* neurons_backend = nullptr;
      ::Backend::CUDA::SpikingSynapses* synapses_backend = nullptr;
    };
  }
}
