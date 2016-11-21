#pragma once

#include "Spike/STDP/MasquelierSTDP.hpp"
#include "STDP.hpp"
#include "Spike/Backend/CUDA/CUDABackend.hpp"
#include "Spike/Helpers/CUDAErrorCheckHelpers.hpp"
#include <cuda.h>
#include <vector_types.h>
#include <curand.h>
#include <curand_kernel.h>

namespace Backend {
  namespace CUDA {
    class MasquelierSTDP : public virtual ::Backend::CUDA::STDPCommon,
                      public ::Backend::MasquelierSTDP {
    public:
      int* index_of_last_afferent_synapse_to_spike = NULL;
      bool* isindexed_ltd_synapse_spike = NULL;
      int* index_of_first_synapse_spiked_after_postneuron = NULL;
      virtual void allocate_device_pointers();
    };
  }
}
