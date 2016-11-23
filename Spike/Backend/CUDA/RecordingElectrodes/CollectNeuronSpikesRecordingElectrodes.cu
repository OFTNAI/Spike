#include "Spike/Backend/CUDA/RecordingElectrodes/CollectNeuronSpikesRecordingElectrodes.hpp"

namespace Backend {
  namespace CUDA {
    CollectNeuronSpikesRecordingElectrodes::~CollectNeuronSpikesRecordingElectrodes() {
      CudaSafeCall(cudaFree(d_neuron_ids_of_stored_spikes_on_device));
      CudaSafeCall(cudaFree(d_total_number_of_spikes_stored_on_device));
      CudaSafeCall(cudaFree(d_time_in_seconds_of_stored_spikes_on_device));
    }

    CollectNeuronSpikesRecordingElectrodes::reset_state() {
      CudaSafeCall(cudaMemset(&(d_total_number_of_spikes_stored_on_device[0]), 0, sizeof(int)));
      CudaSafeCall(cudaMemset(d_neuron_ids_of_stored_spikes_on_device, -1, sizeof(int)*neurons->total_number_of_neurons));
      CudaSafeCall(cudaMemset(d_time_in_seconds_of_stored_spikes_on_device, -1.0f, sizeof(float)*neurons->total_number_of_neurons));
    }

    CollectNeuronSpikesRecordingElectrodes::prepare() {
      printf("TODO Backend::CUDA::CollectNeuronSpikesRecordingElectrodes::prepare\n");
      
      CudaSafeCall(cudaMalloc((void **)&d_neuron_ids_of_stored_spikes_on_device, sizeof(int)*size_of_device_spike_store));
      CudaSafeCall(cudaMalloc((void **)&d_time_in_seconds_of_stored_spikes_on_device, sizeof(float)*size_of_device_spike_store));
      CudaSafeCall(cudaMalloc((void **)&d_total_number_of_spikes_stored_on_device, sizeof(int)));
    }

    CollectNeuronSpikesRecordingElectrodes::copy_spikes_to_front
    (::CollectNeuronSpikesRecordingElectrodes* front) {
      CudaSafeCall(cudaMemcpy((void*)&neuron_ids_of_stored_spikes_on_host[total_number_of_spikes_stored_on_host], 
                              d_neuron_ids_of_stored_spikes_on_device, 
                              (sizeof(int)*total_number_of_spikes_stored_on_device[0]), 
                              cudaMemcpyDeviceToHost));
      CudaSafeCall(cudaMemcpy((void*)&time_in_seconds_of_stored_spikes_on_host[total_number_of_spikes_stored_on_host], 
                              d_time_in_seconds_of_stored_spikes_on_device, 
                              sizeof(float)*total_number_of_spikes_stored_on_device[0], 
                              cudaMemcpyDeviceToHost));
    }

    CollectNeuronSpikesRecordingElectrodes::copy_spike_counts_to_front
    (::CollectNeuronSpikesRecordingElectrodes* front) {
      CudaSafeCall(cudaMemcpy(&(total_number_of_spikes_stored_on_device[0]), &(d_total_number_of_spikes_stored_on_device[0]), (sizeof(int)), cudaMemcpyDeviceToHost));
    }

    CollectNeuronSpikesRecordingElectrodes::collect_spikes_for_timestep
    (::CollectNeuronSpikesRecordingElectrodes* front,
     float current_time_in_seconds) {
      	collect_spikes_for_timestep_kernel<<<neurons->number_of_neuron_blocks_per_grid, neurons->threads_per_block>>>
          (neurons->d_last_spike_time_of_each_neuron,
           d_total_number_of_spikes_stored_on_device,
           d_neuron_ids_of_stored_spikes_on_device,
           d_time_in_seconds_of_stored_spikes_on_device,
           current_time_in_seconds,
           neurons->total_number_of_neurons);

	CudaCheckError();
    }


    // Collect Spikes
    __global__ void collect_spikes_for_timestep_kernel
    (float* d_last_spike_time_of_each_neuron,
     int* d_total_number_of_spikes_stored_on_device,
     int* d_neuron_ids_of_stored_spikes_on_device,
     float* d_time_in_seconds_of_stored_spikes_on_device,
     float current_time_in_seconds,
     size_t total_number_of_neurons){

      int idx = threadIdx.x + blockIdx.x * blockDim.x;
      while (idx < total_number_of_neurons) {

        // If a neuron has fired
        if (d_last_spike_time_of_each_neuron[idx] == current_time_in_seconds) {
          // Increase the number of spikes stored
          // NOTE: atomicAdd return value is actually original (atomic) value BEFORE incrementation!
          //		- So first value is actually 0 not 1!!!
          int i = atomicAdd(&d_total_number_of_spikes_stored_on_device[0], 1);
          __syncthreads();

          // In the location, add the id and the time
          d_neuron_ids_of_stored_spikes_on_device[i] = idx;
          d_time_in_seconds_of_stored_spikes_on_device[i] = current_time_in_seconds;
        }
        idx += blockDim.x * gridDim.x;
      }
      __syncthreads();
    }
  }
}
