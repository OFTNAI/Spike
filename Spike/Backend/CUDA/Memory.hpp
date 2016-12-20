#pragma once

#include <cstddef>

#include "Spike/Backend/CUDA/Helpers/ErrorCheck.hpp"

namespace Backend {
  namespace CUDA {
    inline size_t memory_total_bytes(Context* ctx = _global_ctx) {
      size_t tmp_free, tmp_total;
      cudaError_t cuda_status = cudaMemGetInfo(&tmp_free, &tmp_total) ;
      if ( cudaSuccess != cuda_status ){
        printf("Error: cudaMemGetInfo fails, %s \n", cudaGetErrorString(cuda_status) );
        exit(1);
      }
      return tmp_total;
    }

    inline size_t memory_free_bytes(Context* ctx = _global_ctx) {
      size_t tmp_free, tmp_total;
      cudaError_t cuda_status = cudaMemGetInfo(&tmp_free, &tmp_total) ;
      if ( cudaSuccess != cuda_status ){
        printf("Error: cudaMemGetInfo fails, %s \n", cudaGetErrorString(cuda_status) );
        exit(1);
      }
      return tmp_free;
    }
  }
}

