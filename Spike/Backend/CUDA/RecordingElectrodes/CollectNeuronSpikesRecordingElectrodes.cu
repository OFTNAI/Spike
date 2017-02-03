// -*- mode: c++ -*-
#include "Spike/Backend/CUDA/RecordingElectrodes/CollectNeuronSpikesRecordingElectrodes.hpp"

SPIKE_EXPORT_BACKEND_TYPE(CUDA, CollectNeuronSpikesRecordingElectrodes);

namespace Backend {
  namespace CUDA {
    CollectNeuronSpikesRecordingElectrodes::~CollectNeuronSpikesRecordingElectrodes() {
      CudaSafeCall(cudaFree(neuron_ids_of_stored_spikes_on_device));
      CudaSafeCall(cudaFree(total_number_of_spikes_stored_on_device));
      CudaSafeCall(cudaFree(time_in_seconds_of_stored_spikes_on_device));
    }

    void CollectNeuronSpikesRecordingElectrodes::reset_state() {
      RecordingElectrodes::reset_state();

      CudaSafeCall(cudaMemset(&(total_number_of_spikes_stored_on_device[0]), 0, sizeof(int)));
      CudaSafeCall(cudaMemcpy(neuron_ids_of_stored_spikes_on_device, frontend()->reset_neuron_ids, sizeof(int)*frontend()->size_of_device_spike_store, cudaMemcpyHostToDevice));
      CudaSafeCall(cudaMemcpy(time_in_seconds_of_stored_spikes_on_device, frontend()->reset_neuron_times, sizeof(float)*frontend()->size_of_device_spike_store, cudaMemcpyHostToDevice));
    }

    void CollectNeuronSpikesRecordingElectrodes::prepare() {
      RecordingElectrodes::prepare();

      CudaSafeCall(cudaMalloc((void **)&neuron_ids_of_stored_spikes_on_device, sizeof(int)*frontend()->size_of_device_spike_store));
      CudaSafeCall(cudaMalloc((void **)&time_in_seconds_of_stored_spikes_on_device, sizeof(float)*frontend()->size_of_device_spike_store));
      CudaSafeCall(cudaMalloc((void **)&total_number_of_spikes_stored_on_device, sizeof(int)));
    
      reset_state();
    }

    void CollectNeuronSpikesRecordingElectrodes::copy_spikes_to_front() {
      CudaSafeCall(cudaMemcpy((void*)&frontend()->neuron_ids_of_stored_spikes_on_host[frontend()->total_number_of_spikes_stored_on_host], 
                              neuron_ids_of_stored_spikes_on_device, 
                              (sizeof(int)*frontend()->total_number_of_spikes_stored_on_device[0]), 
                              cudaMemcpyDeviceToHost));
      CudaSafeCall(cudaMemcpy((void*)&frontend()->time_in_seconds_of_stored_spikes_on_host[frontend()->total_number_of_spikes_stored_on_host], 
                              time_in_seconds_of_stored_spikes_on_device, 
                              sizeof(float)*frontend()->total_number_of_spikes_stored_on_device[0], 
                              cudaMemcpyDeviceToHost));
    }

    void CollectNeuronSpikesRecordingElectrodes::copy_spike_counts_to_front() {
      CudaSafeCall(cudaMemcpy(&(frontend()->total_number_of_spikes_stored_on_device[0]), &(total_number_of_spikes_stored_on_device[0]), (sizeof(int)), cudaMemcpyDeviceToHost));
    }

    void CollectNeuronSpikesRecordingElectrodes::collect_spikes_for_timestep
    (float current_time_in_seconds) {
      collect_spikes_for_timestep_kernel<<<neurons_backend->number_of_neuron_blocks_per_grid, neurons_backend->threads_per_block>>>
        (neurons_backend->last_spike_time_of_each_neuron,
         total_number_of_spikes_stored_on_device,
         neuron_ids_of_stored_spikes_on_device,
         time_in_seconds_of_stored_spikes_on_device,
         current_time_in_seconds,
         neurons_frontend->total_number_of_neurons);

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
