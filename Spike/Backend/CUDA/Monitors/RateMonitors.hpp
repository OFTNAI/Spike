#pragma once

#include "Spike/Monitors/RateMonitors.hpp"
#include "Monitors.hpp"
#include "Spike/Backend/CUDA/CUDABackend.hpp"
#include <cuda.h>
#include <vector_types.h>
#include <curand.h>
#include <curand_kernel.h>

namespace Backend {
  namespace CUDA {
    class RateMonitors :
      public virtual ::Backend::CUDA::Monitors,
      public virtual ::Backend::RateMonitors {
    public:
      ~RateMonitors() override;
      SPIKE_MAKE_BACKEND_CONSTRUCTOR(RateMonitors);
      using ::Backend::RateMonitors::frontend;
      
      void prepare() override;
      void reset_state() override;

      void allocate_pointers_for_spike_count(); // Not virtual

      void add_spikes_to_per_neuron_spike_count
      (float current_time_in_seconds) override;

      int * per_neuron_spike_counts = nullptr;
    };

    __global__ void add_spikes_to_per_neuron_spike_count_kernel
    (float* d_last_spike_time_of_each_neuron,
     int* d_per_neuron_spike_counts,
     float current_time_in_seconds,
     size_t total_number_of_neurons);
  }
}
