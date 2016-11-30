#pragma once

#include "Spike/Helpers/RandomStateManager.hpp"
#include "Spike/Backend/CUDA/CUDABackend.hpp"

#include <cuda.h>
#include <vector_types.h>
#include <curand.h>
#include <curand_kernel.h>

namespace Backend {
  namespace CUDA {
    class RandomStateManager : public ::Backend::RandomStateManager {
    public:
      curandState_t* d_states = nullptr;
      dim3 threads_per_block;
      dim3 block_dimensions;

      MAKE_BACKEND_CONSTRUCTOR(RandomStateManager);

      void setup_random_states(int threads_per_blocks_x = 128, int number_of_blocks_x = 64, int seed = 1);

      virtual void prepare();
    };

    __global__ void generate_random_states_kernel(unsigned int seed, curandState_t* d_states, size_t total_number);
  }
}
