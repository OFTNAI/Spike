#include "Memory.hpp"
#include "ErrorCheck.hpp"

namespace Backend {
  namespace CUDA {
    std::size_t MemoryManager::total_bytes() const {
      size_t tmp_free, tmp_total;
      cudaError_t cuda_status = cudaMemGetInfo(&tmp_free, &tmp_total) ;
      if ( cudaSuccess != cuda_status ){
        printf("Error: cudaMemGetInfo fails, %s \n", cudaGetErrorString(cuda_status) );
        exit(1);
      }
      return tmp_total;
    }

    std::size_t MemoryManager::free_bytes() const {
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

SPIKE_EXPORT_BACKEND_TYPE(CUDA, MemoryManager);
