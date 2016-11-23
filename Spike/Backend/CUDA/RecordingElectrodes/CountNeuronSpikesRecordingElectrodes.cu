#include "Spike/Backend/CUDA/RecordingElectrodes/CountNeuronSpikesRecordingElectrodes.hpp"

namespace Backend {
  namespace CUDA {
    CountNeuronSpikesRecordingElectrodes::~CountNeuronSpikesRecordingElectrodes() {
      CudaSafeCall(cudaFree(d_per_neuron_spike_counts));
    }

    CountNeuronSpikesRecordingElectrodes::reset_state() {
      CudaSafeCall(cudaMemset(d_per_neuron_spike_counts, 0, sizeof(int) * neurons->total_number_of_neurons));
    }

    CountNeuronSpikesRecordingElectrodes::prepare() {
    }

    CountNeuronSpikesRecordingElectrodes::allocate_pointers_for_spike_count() {
      CudaSafeCall(cudaMalloc((void **)&d_per_neuron_spike_counts, sizeof(int) * neurons->total_number_of_neurons));
    }

    CountNeuronSpikesRecordingElectrodes::add_spikes_to_per_neuron_spike_count
    (::CountNeuronSpikesRecordingElectrodes* front,
     float current_time_in_seconds) {
      add_spikes_to_per_neuron_spike_count_kernel<<<neurons->number_of_neuron_blocks_per_grid, neurons->threads_per_block>>>
        (neurons->d_last_spike_time_of_each_neuron,
         d_per_neuron_spike_counts,
         current_time_in_seconds,
         neurons->total_number_of_neurons);
	CudaCheckError();
    }

    __global__ void add_spikes_to_per_neuron_spike_count_kernel
    (float* d_last_spike_time_of_each_neuron,
     int* d_per_neuron_spike_counts,
     float current_time_in_seconds,
     size_t total_number_of_neurons) {

      int idx = threadIdx.x + blockIdx.x * blockDim.x;
      while (idx < total_number_of_neurons) {

        if (d_last_spike_time_of_each_neuron[idx] == current_time_in_seconds) {
          atomicAdd(&d_per_neuron_spike_counts[idx], 1);
        }

        // if (idx == 1000) printf("d_per_neuron_spike_counts[idx]: %d\n", d_per_neuron_spike_counts[idx]);
        idx += blockDim.x * gridDim.x;
      }
    }
  }
}

