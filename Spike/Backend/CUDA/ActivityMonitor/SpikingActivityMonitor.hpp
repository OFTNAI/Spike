#pragma once

#include "Spike/ActivityMonitor/SpikingActivityMonitor.hpp"
#include "ActivityMonitor.hpp"
#include "Spike/Backend/CUDA/CUDABackend.hpp"
#include <cuda.h>
#include <vector_types.h>
#include <curand.h>
#include <curand_kernel.h>

namespace Backend {
  namespace CUDA {
    class SpikingActivityMonitor :
      public virtual ::Backend::CUDA::ActivityMonitor,
      public virtual ::Backend::SpikingActivityMonitor {
    public:
      ~SpikingActivityMonitor() override;
      SPIKE_MAKE_BACKEND_CONSTRUCTOR(SpikingActivityMonitor);
      using ::Backend::SpikingActivityMonitor::frontend;

      void prepare() override;
      void reset_state() override;

      void copy_spikes_to_front() override;
      void copy_spikecount_to_front() override;

      void collect_spikes_for_timestep(float current_time_in_seconds, float timestep) override;

      int* neuron_ids_of_stored_spikes_on_device = nullptr;
      int* total_number_of_spikes_stored_on_device = nullptr;
      float* time_in_seconds_of_stored_spikes_on_device = nullptr;
    };

    __global__ void collect_spikes_for_timestep_kernel
    (float* d_last_spike_time_of_each_neuron,
     int* d_total_number_of_spikes_stored_on_device,
     int* d_neuron_ids_of_stored_spikes_on_device,
     float* d_time_in_seconds_of_stored_spikes_on_device,
     float current_time_in_seconds,
     float timestep,
     size_t total_number_of_neurons);
  }
}
