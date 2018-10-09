#pragma once

#include "Spike/ActivityMonitor/RateActivityMonitor.hpp"
#include "ActivityMonitor.hpp"
#include "Spike/Backend/CUDA/CUDABackend.hpp"
#include <cuda.h>
#include <vector_types.h>
#include <curand.h>
#include <curand_kernel.h>

namespace Backend {
  namespace CUDA {
    class RateActivityMonitor :
      public virtual ::Backend::CUDA::ActivityMonitor,
      public virtual ::Backend::RateActivityMonitor {
    public:
      ~RateActivityMonitor() override;
      SPIKE_MAKE_BACKEND_CONSTRUCTOR(RateActivityMonitor);
      using ::Backend::RateActivityMonitor::frontend;
      
      int * per_neuron_spike_counts = nullptr;
      
      void prepare() override;
      void reset_state() override;

      void allocate_pointers_for_spike_count(); // Not virtual

      void copy_spike_count_to_host() override;
      void add_spikes_to_per_neuron_spike_count(float current_time_in_seconds, float timestep) override;

    };

    __global__ void add_spikes_to_per_neuron_spike_count_kernel
    (spiking_neurons_data_struct* neuron_data,
     int* d_per_neuron_spike_counts,
     float current_time_in_seconds,
     float timestep,
     int timestep_grouping,
     size_t total_number_of_neurons);
  }
}
