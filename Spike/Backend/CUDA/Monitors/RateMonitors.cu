// -*- mode: c++ -*-
#include "Spike/Backend/CUDA/Monitors/RateMonitors.hpp"

SPIKE_EXPORT_BACKEND_TYPE(CUDA, RateMonitors);

namespace Backend {
  namespace CUDA {
    RateMonitors::~RateMonitors() {
      CudaSafeCall(cudaFree(per_neuron_spike_counts));
    }

    void RateMonitors::reset_state() {
      Monitors::reset_state();

      CudaSafeCall(cudaMemset(per_neuron_spike_counts, 0, sizeof(int) * frontend()->neurons->total_number_of_neurons));
    }

    void RateMonitors::prepare() {
      Monitors::prepare();

      allocate_pointers_for_spike_count();
    }

    void RateMonitors::allocate_pointers_for_spike_count() {
      CudaSafeCall(cudaMalloc((void **)&per_neuron_spike_counts,
                              sizeof(int) * frontend()->neurons->total_number_of_neurons));
    }

    void RateMonitors::add_spikes_to_per_neuron_spike_count
    (float current_time_in_seconds) {
      add_spikes_to_per_neuron_spike_count_kernel<<<neurons_backend->number_of_neuron_blocks_per_grid, neurons_backend->threads_per_block>>>
        (neurons_backend->last_spike_time_of_each_neuron,
         per_neuron_spike_counts,
         current_time_in_seconds,
         frontend()->neurons->total_number_of_neurons);
         CudaCheckError();
    }

    __global__ void add_spikes_to_per_neuron_spike_count_kernel
    (float* d_last_spike_time_of_each_neuron,
     int* d_per_neuron_spike_counts,
     float current_time_in_seconds,
     size_t total_number_of_neurons) {

      int idx = threadIdx.x + blockIdx.x * blockDim.x;
      while (idx < total_number_of_neurons) {

        if (d_last_spike_time_of_each_neuron[idx] >= current_time_in_seconds) {
          atomicAdd(&d_per_neuron_spike_counts[idx], 1);
        }

        idx += blockDim.x * gridDim.x;
      }
    }
  }
}

