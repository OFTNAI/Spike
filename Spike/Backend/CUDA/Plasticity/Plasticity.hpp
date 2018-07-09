#pragma once

#include "Spike/Plasticity/Plasticity.hpp"
#include "Spike/Backend/CUDA/CUDABackend.hpp"

#include <cuda.h>
#include <vector_types.h>
#include <curand.h>
#include <curand_kernel.h>

namespace Backend {
  namespace CUDA {
    class Plasticity : public virtual ::Backend::Plasticity {
    public:
      ~Plasticity() override;
      using ::Backend::Plasticity::frontend;

      void prepare() override;
      void reset_state() override;
      void allocate_device_pointers();

    };
  }
}
