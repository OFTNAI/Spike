// -*- mode: c++ -*-
#include "Spike/Backend/CUDA/Neurons/GeneratorInputSpikingNeurons.hpp"

SPIKE_EXPORT_BACKEND_TYPE(CUDA, GeneratorInputSpikingNeurons)

namespace Backend {
  namespace CUDA {
    GeneratorInputSpikingNeurons::~GeneratorInputSpikingNeurons() {
      CudaSafeCall(cudaFree(neuron_ids_for_stimulus));
      CudaSafeCall(cudaFree(spike_times_for_stimulus));
    }
    
    // Allocate device pointers for the longest stimulus so that they do not need to be replaced
    void GeneratorInputSpikingNeurons::allocate_device_pointers() {
      CudaSafeCall(cudaMalloc((void **)&neuron_ids_for_stimulus, sizeof(int)*frontend()->length_of_longest_stimulus));
      CudaSafeCall(cudaMalloc((void **)&spike_times_for_stimulus, sizeof(float)*frontend()->length_of_longest_stimulus));
    }

    void GeneratorInputSpikingNeurons::prepare() {
      InputSpikingNeurons::prepare();
      allocate_device_pointers();
    }

    void GeneratorInputSpikingNeurons::reset_state() {
      InputSpikingNeurons::reset_state();

      CudaSafeCall(cudaMemcpy(neuron_ids_for_stimulus,
                              frontend()->neuron_id_matrix_for_stimuli[frontend()->current_stimulus_index],
                              sizeof(int)*frontend()->number_of_spikes_in_stimuli[frontend()->current_stimulus_index],
                              cudaMemcpyHostToDevice));
      CudaSafeCall(cudaMemcpy(spike_times_for_stimulus,
                              frontend()->spike_times_matrix_for_stimuli[frontend()->current_stimulus_index],
                              sizeof(float)*frontend()->number_of_spikes_in_stimuli[frontend()->current_stimulus_index],
                              cudaMemcpyHostToDevice));
      num_spikes_in_current_stimulus = frontend()->number_of_spikes_in_stimuli[frontend()->current_stimulus_index];
    }

    void GeneratorInputSpikingNeurons::state_update(float current_time_in_seconds, float timestep) {
      if ((frontend()->temporal_lengths_of_stimuli[frontend()->current_stimulus_index] +  timestep) > current_time_in_seconds){
        check_for_generator_spikes_kernel<<<number_of_neuron_blocks_per_grid, threads_per_block>>>
          (neuron_ids_for_stimulus,
           spike_times_for_stimulus,
           last_spike_time_of_each_neuron,
           current_time_in_seconds,
           timestep,
           num_spikes_in_current_stimulus);

        CudaCheckError();
      }
    }


    __global__ void check_for_generator_spikes_kernel(int *d_neuron_ids_for_stimulus,
                                                      float *d_spike_times_for_stimulus,
                                                      float* d_last_spike_time_of_each_neuron,
                                                      float current_time_in_seconds,
                                                      float timestep,
                                                      size_t number_of_spikes_in_stimulus){

      // // Get thread IDs
      int idx = threadIdx.x + blockIdx.x * blockDim.x;
      while (idx < number_of_spikes_in_stimulus) {
        if (fabs(current_time_in_seconds - d_spike_times_for_stimulus[idx]) < 0.5 * timestep) {
          d_last_spike_time_of_each_neuron[d_neuron_ids_for_stimulus[idx]] = current_time_in_seconds;
        }

        idx += blockDim.x * gridDim.x;
      }
      __syncthreads();
    }
  }
}
