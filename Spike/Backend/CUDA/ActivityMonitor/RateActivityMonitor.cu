// -*- mode: c++ -*-
#include "Spike/Backend/CUDA/ActivityMonitor/RateActivityMonitor.hpp"

SPIKE_EXPORT_BACKEND_TYPE(CUDA, RateActivityMonitor);

namespace Backend {
  namespace CUDA {
    RateActivityMonitor::~RateActivityMonitor() {
      CudaSafeCall(cudaFree(per_neuron_spike_counts));
    }

    void RateActivityMonitor::reset_state() {
      ActivityMonitor::reset_state();

      CudaSafeCall(cudaMemset(per_neuron_spike_counts, 0, sizeof(int) * frontend()->neurons->total_number_of_neurons));
    }

    void RateActivityMonitor::prepare() {
      ActivityMonitor::prepare();
      neurons_frontend = frontend()->neurons;
      neurons_backend =
        dynamic_cast<::Backend::CUDA::SpikingNeurons*>(neurons_frontend->backend());
      allocate_pointers_for_spike_count();
    }

    void RateActivityMonitor::allocate_pointers_for_spike_count() {
      CudaSafeCall(cudaMalloc((void **)&per_neuron_spike_counts,
                              sizeof(int) * frontend()->neurons->total_number_of_neurons));
    }

    void RateActivityMonitor::copy_spike_count_to_host(){
      CudaSafeCall(cudaMemcpy((void*)&(frontend()->per_neuron_spike_counts[0]), 
                              per_neuron_spike_counts, 
                              sizeof(int)*frontend()->neurons->total_number_of_neurons,
                              cudaMemcpyDeviceToHost));
    }

    void RateActivityMonitor::add_spikes_to_per_neuron_spike_count
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

