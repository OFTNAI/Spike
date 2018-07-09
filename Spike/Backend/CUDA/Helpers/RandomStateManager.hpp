#pragma once

#include "Spike/Helpers/RandomStateManager.hpp"
#include "Spike/Backend/CUDA/CUDABackend.hpp"

#include <cuda.h>
#include <vector_types.h>
#include <curand.h>
#include <curand_kernel.h>

namespace Backend {
  namespace CUDA {
    class RandomStateManager : public virtual ::Backend::RandomStateManager {
    public:
      curandState_t* states = nullptr;
      dim3 threads_per_block;
      dim3 block_dimensions;
      int total_number_of_states = 0;

      ~RandomStateManager() override;
      SPIKE_MAKE_BACKEND_CONSTRUCTOR(RandomStateManager);
      using ::Backend::RandomStateManager::frontend;

      void setup_random_states(int threads_per_blocks_x = 128, int number_of_blocks_x = 64, int seed = 1);

      void prepare() override;
    };

    __global__ void generate_random_states_kernel(unsigned int seed, curandState_t* d_states, size_t total_number);
  }
}
