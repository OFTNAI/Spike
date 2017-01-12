// -*- mode: c++ -*-
#include "RandomStateManager.hpp"
#include "Spike/Helpers/TimerWithMessages.hpp"
#include "Spike/Helpers/TerminalHelpers.hpp"

SPIKE_EXPORT_BACKEND_TYPE(CUDA, RandomStateManager);

namespace Backend {
  namespace CUDA {
    RandomStateManager::~RandomStateManager() {
      CudaSafeCall(cudaFree(states));
    }

    void RandomStateManager::setup_random_states(int threads_per_blocks_x, int number_of_blocks_x, int seed) {
      // TimerWithMessages * set_up_random_states_timer = new TimerWithMessages("Setting up random states for RandomStateManager...\n");

      threads_per_block = dim3(threads_per_blocks_x);
      block_dimensions = dim3(number_of_blocks_x);
      total_number_of_states = threads_per_blocks_x * number_of_blocks_x;

      // In case it has already been allocated
      if (states) {
        CudaSafeCall(cudaFree(states));
        states = nullptr;
      }

      // Allocate the random states
      CudaSafeCall(cudaMalloc((void**) &states, sizeof(curandState_t)*threads_per_blocks_x*number_of_blocks_x));
      generate_random_states_kernel<<<block_dimensions, threads_per_block>>>(seed, states, threads_per_blocks_x * number_of_blocks_x);
      CudaCheckError();

      // set_up_random_states_timer->stop_timer_and_log_time_and_message("Random states set up...", true);
    }

    void RandomStateManager::prepare() {
      setup_random_states();
    }

    __global__ void generate_random_states_kernel(unsigned int seed, curandState_t* d_states, size_t total_number) {
      int idx = threadIdx.x + blockIdx.x * blockDim.x;
      // int idx_g = idx;
      if (idx < total_number) {
        curand_init(seed, // the seed can be the same for each core, here we pass the time in from the CPU
                    idx,  // the sequence number should be different for each core (unless you want all
                    // cores to get the same sequence of numbers for some reason - use thread id!
                    0,    // the offset is how much extra we advance in the sequence for each call; can be 0
                    &d_states[idx]);

        __syncthreads();
      }
    }
  }
}
