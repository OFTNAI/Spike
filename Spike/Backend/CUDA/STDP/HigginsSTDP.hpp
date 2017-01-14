#pragma once

#include "Spike/STDP/HigginsSTDP.hpp"
#include "STDP.hpp"
#include "Spike/Backend/CUDA/CUDABackend.hpp"
#include <cuda.h>
#include <vector_types.h>
#include <curand.h>
#include <curand_kernel.h>

namespace Backend {
  namespace CUDA {
    class HigginsSTDP : public virtual ::Backend::CUDA::STDP,
                        public virtual ::Backend::HigginsSTDP {
    public:
      ~HigginsSTDP() override = default;
      SPIKE_MAKE_BACKEND_CONSTRUCTOR(HigginsSTDP);
      using ::Backend::HigginsSTDP::frontend;

      void prepare() override;
      void reset_state() override;

      // void allocate_device_pointers();
      void apply_ltp_to_synapse_weights(float current_time_in_seconds) override;
      void apply_ltd_to_synapse_weights(float current_time_in_seconds) override;
    };

    // Kernels to carry out LTP/LTD
    __global__ void izhikevich_apply_ltd_to_synapse_weights_kernel
    (float* d_time_of_last_spike_to_reach_synapse,
     float* d_synaptic_efficacies_or_weights,
     bool* d_stdp,
     float* d_last_spike_time_of_each_neuron,
     int* d_postsyns,
     float currtime,
     struct higgins_stdp_parameters_struct stdp_vars,
     size_t total_number_of_synapse);

    __global__ void izhikevich_apply_ltp_to_synapse_weights_kernel
    (int* d_postsyns,
     float* d_last_spike_time_of_each_neuron,
     bool* d_stdp,
     float* d_time_of_last_spike_to_reach_synapse,
     float* d_synaptic_efficacies_or_weights,
     struct higgins_stdp_parameters_struct stdp_vars,
     float currtime,
     size_t total_number_of_synapse);
  }
}
