// -*- mode: c++ -*-
#include "Spike/Backend/CUDA/Neurons/GeneratorInputSpikingNeurons.hpp"

namespace Backend {
  namespace CUDA {
    GeneratorInputSpikingNeurons::~GeneratorInputSpikingNeurons() {
      CudaSafeCall(cudaFree(neuron_ids_for_stimulus));
      CudaSafeCall(cudaFree(spike_times_for_stimulus));
    }
    
    // Allocate device pointers for the longest stimulus so that they do not need to be replaced
    void GeneratorInputSpikingNeurons::allocate_device_pointers(int maximum_axonal_delay_in_timesteps, bool high_fidelity_spike_storage) {
      InputSpikingNeurons::allocate_device_pointers(maximum_axonal_delay_in_timesteps, high_fidelity_spike_storage);

      CudaSafeCall(cudaMalloc((void **)&neuron_ids_for_stimulus, sizeof(int)*frontend()->length_of_longest_stimulus));
      CudaSafeCall(cudaMalloc((void **)&spike_times_for_stimulus, sizeof(float)*frontend()->length_of_longest_stimulus));
    }

    void GeneratorInputSpikingNeurons::reset_state() {
      CudaSafeCall(cudaMemcpy(neuron_ids_for_stimulus,
                              frontend()->neuron_id_matrix_for_stimuli[frontend()->current_stimulus_index],
                              sizeof(int)*frontend()->number_of_spikes_in_stimuli[frontend()->current_stimulus_index],
                              cudaMemcpyHostToDevice));
      CudaSafeCall(cudaMemcpy(spike_times_for_stimulus,
                              frontend()->spike_times_matrix_for_stimuli[frontend()->current_stimulus_index],
                              sizeof(float)*frontend()->number_of_spikes_in_stimuli[frontend()->current_stimulus_index],
                              cudaMemcpyHostToDevice));
    }

    void GeneratorInputSpikingNeurons::check_for_neuron_spikes(float current_time_in_seconds, float timestep) {
      check_for_generator_spikes_kernel<<<number_of_neuron_blocks_per_grid, threads_per_block>>>
        (neuron_ids_for_stimulus,
         spike_times_for_stimulus,
         last_spike_time_of_each_neuron,
         bitarray_of_neuron_spikes,
         frontend()->bitarray_length,
         frontend()->bitarray_maximum_axonal_delay_in_timesteps,
         current_time_in_seconds,
         timestep,
         frontend()->number_of_spikes_in_stimuli[frontend()->current_stimulus_index],
         frontend()->high_fidelity_spike_flag);

      CudaCheckError();
    }
  }

  __global__ void check_for_generator_spikes_kernel(int *d_neuron_ids_for_stimulus,
                                                    float *d_spike_times_for_stimulus,
                                                    float* d_last_spike_time_of_each_neuron,
                                                    unsigned char* d_bitarray_of_neuron_spikes,
                                                    int bitarray_length,
                                                    int bitarray_maximum_axonal_delay_in_timesteps,
                                                    float current_time_in_seconds,
                                                    float timestep,
                                                    size_t number_of_spikes_in_stimulus,
                                                    bool high_fidelity_spike_flag) {

    // // Get thread IDs
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    while (idx < number_of_spikes_in_stimulus) {
      if (fabs(current_time_in_seconds - d_spike_times_for_stimulus[idx]) < 0.5 * timestep) {
        __syncthreads();
        d_last_spike_time_of_each_neuron[d_neuron_ids_for_stimulus[idx]] = current_time_in_seconds;

        if (high_fidelity_spike_flag){
          // Get start of the given neuron's bits
          int neuron_id_spike_store_start = d_neuron_ids_for_stimulus[idx] * bitarray_length;
          // Get offset depending upon the current timestep
          int offset_index = (int)(round((float)(current_time_in_seconds / timestep))) % bitarray_maximum_axonal_delay_in_timesteps;
          int offset_byte = offset_index / 8;
          int offset_bit_pos = offset_index - (8 * offset_byte);
          // Get the specific position at which we should be putting the current value
          unsigned char byte = d_bitarray_of_neuron_spikes[neuron_id_spike_store_start + offset_byte];
          // Set the specific bit in the byte to on 
          byte |= (1 << offset_bit_pos);
          // Assign the byte
          d_bitarray_of_neuron_spikes[neuron_id_spike_store_start + offset_byte] = byte;
        }
      } else {
        // High fidelity spike storage
        if (high_fidelity_spike_flag){
          // Get start of the given neuron's bits
          int neuron_id_spike_store_start = d_neuron_ids_for_stimulus[idx] * bitarray_length;
          // Get offset depending upon the current timestep
          int offset_index = (int)(round((float)(current_time_in_seconds / timestep))) % bitarray_maximum_axonal_delay_in_timesteps;
          int offset_byte = offset_index / 8;
          int offset_bit_pos = offset_index - (8 * offset_byte);
          // Get the specific position at which we should be putting the current value
          unsigned char byte = d_bitarray_of_neuron_spikes[neuron_id_spike_store_start + offset_byte];
          // Set the specific bit in the byte to on 
          byte &= ~(1 << offset_bit_pos);
          // Assign the byte
          d_bitarray_of_neuron_spikes[neuron_id_spike_store_start + offset_byte] = byte;
        }
      }

      idx += blockDim.x * gridDim.x;
    }
    __syncthreads();
  }
}
