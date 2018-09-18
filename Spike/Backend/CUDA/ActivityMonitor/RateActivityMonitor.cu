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
    (float current_time_in_seconds, float timestep) {
      add_spikes_to_per_neuron_spike_count_kernel<<<neurons_backend->number_of_neuron_blocks_per_grid, neurons_backend->threads_per_block>>>
        (neurons_backend->d_neuron_data,
         per_neuron_spike_counts,
         current_time_in_seconds,
         timestep,
         frontend()->model->timestep_grouping,
         frontend()->neurons->total_number_of_neurons);
         CudaCheckError();
    }

    __global__ void add_spikes_to_per_neuron_spike_count_kernel
    (spiking_neurons_data_struct* neuron_data,
     int* d_per_neuron_spike_counts,
     float current_time_in_seconds,
     float timestep,
     int timestep_grouping,
     size_t total_number_of_neurons) {
      
      int idx = threadIdx.x + blockIdx.x * blockDim.x;
      int bufsize = neuron_data->neuron_spike_time_bitbuffer_bytesize[0];
      while (idx < total_number_of_neurons) {
        for (int g=0; g < timestep_grouping; g++){
          int bitloc = ((int)roundf(current_time_in_seconds / timestep) + g) % (8*bufsize);
          // If a neuron has fired
          if (neuron_data->neuron_spike_time_bitbuffer[idx*bufsize + (bitloc / 8)] & (1 << (bitloc % 8))){
            atomicAdd(&d_per_neuron_spike_counts[idx], 1);
          }
        }

        idx += blockDim.x * gridDim.x;
      }
    }

  }
}

