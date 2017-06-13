// -*- mode: c++ -*-
#include "Spike/Backend/CUDA/Neurons/SpikingNeurons.hpp"

// SPIKE_EXPORT_BACKEND_TYPE(CUDA, SpikingNeurons);

namespace Backend {
  namespace CUDA {
    SpikingNeurons::~SpikingNeurons() {
      CudaSafeCall(cudaFree(last_spike_time_of_each_neuron));
      CudaSafeCall(cudaFree(membrane_potentials_v));
      CudaSafeCall(cudaFree(thresholds_for_action_potential_spikes));
      CudaSafeCall(cudaFree(resting_potentials));
      if (bitarray_of_neuron_spikes)
      	CudaSafeCall(cudaFree(bitarray_of_neuron_spikes));
    }

    void SpikingNeurons::allocate_device_pointers(int maximum_axonal_delay_in_timesteps, bool high_fidelity_spike_storage) {
      CudaSafeCall(cudaMalloc((void **)&last_spike_time_of_each_neuron, sizeof(float)*frontend()->total_number_of_neurons));
      CudaSafeCall(cudaMalloc((void **)&membrane_potentials_v, sizeof(float)*frontend()->total_number_of_neurons));
      CudaSafeCall(cudaMalloc((void **)&thresholds_for_action_potential_spikes, sizeof(float)*frontend()->total_number_of_neurons));
      CudaSafeCall(cudaMalloc((void **)&resting_potentials, sizeof(float)*frontend()->total_number_of_neurons));

      if (high_fidelity_spike_storage){
        // Create bit array of correct length
        CudaSafeCall(cudaMalloc((void **)&bitarray_of_neuron_spikes, sizeof(unsigned char)*frontend()->bitarray_length*frontend()->total_number_of_neurons));
      }
    }

    void SpikingNeurons::copy_constants_to_device() {
      CudaSafeCall(cudaMemcpy(thresholds_for_action_potential_spikes, frontend()->thresholds_for_action_potential_spikes, sizeof(float)*frontend()->total_number_of_neurons, cudaMemcpyHostToDevice));
      CudaSafeCall(cudaMemcpy(resting_potentials, frontend()->after_spike_reset_membrane_potentials_c, sizeof(float)*frontend()->total_number_of_neurons, cudaMemcpyHostToDevice));
    }

    void SpikingNeurons::prepare() {
      Neurons::prepare();
      allocate_device_pointers(context->params.maximum_axonal_delay_in_timesteps, context->params.high_fidelity_spike_storage);
      copy_constants_to_device();
    }

    void SpikingNeurons::reset_state() {
      Neurons::reset_state();

      // Set last spike times to -1000 so that the times do not affect current simulation.
      float* tmp_last_spike_times;
      tmp_last_spike_times = (float*)malloc(sizeof(float)*frontend()->total_number_of_neurons);
      for (int i=0; i < frontend()->total_number_of_neurons; i++){
        tmp_last_spike_times[i] = -1000.0f;
      }

      CudaSafeCall(cudaMemcpy(last_spike_time_of_each_neuron,
                              tmp_last_spike_times,
                              frontend()->total_number_of_neurons*sizeof(float),
                              cudaMemcpyHostToDevice));
      CudaSafeCall(cudaMemcpy(membrane_potentials_v,
                              frontend()->after_spike_reset_membrane_potentials_c,
                              sizeof(float)*frontend()->total_number_of_neurons,
                              cudaMemcpyHostToDevice));

      if (frontend()->high_fidelity_spike_flag) {
        CudaSafeCall(cudaMemcpy(bitarray_of_neuron_spikes,
                                frontend()->bitarray_of_neuron_spikes,
                                sizeof(unsigned char)*frontend()->bitarray_length*frontend()->total_number_of_neurons,
                                cudaMemcpyHostToDevice));
      }

      // Free tmp_last_spike_times
      free (tmp_last_spike_times);
    }

    void SpikingNeurons::state_update(float current_time_in_seconds, float timestep) {
      if (frontend()->high_fidelity_spike_flag){
        high_fidelity_check_for_neuron_spikes_kernel<<<number_of_neuron_blocks_per_grid, threads_per_block>>>
          (membrane_potentials_v,
           thresholds_for_action_potential_spikes,
           resting_potentials,
           last_spike_time_of_each_neuron,
           bitarray_of_neuron_spikes,
           frontend()->bitarray_length,
           frontend()->bitarray_maximum_axonal_delay_in_timesteps,
           current_time_in_seconds,
           timestep,
           frontend()->total_number_of_neurons);
      } else {
      check_for_neuron_spikes_kernel<<<number_of_neuron_blocks_per_grid, threads_per_block>>>
        (membrane_potentials_v,
         thresholds_for_action_potential_spikes,
         resting_potentials,
         last_spike_time_of_each_neuron,
         current_time_in_seconds,
         frontend()->total_number_of_neurons);
  
      CudaCheckError();
      }
    }

    __device__ void check_for_neuron_spikes_kernel(float& membrane_potentials_v,
                                                   float thresholds_for_action_potential_spikes,
                                                   float resting_potentials,
                                                   float& last_spike_time_of_each_neuron,
                                                   float current_time_in_seconds) {
      if (membrane_potentials_v >= thresholds_for_action_potential_spikes) {

        // Set current time as last spike time of neuron
        last_spike_time_of_each_neuron = current_time_in_seconds;

        // Reset membrane potential
        membrane_potentials_v = resting_potentials;

      }

    }

    __global__ void high_fidelity_check_for_neuron_spikes_kernel(float *membrane_potentials_v,
                                                   float *thresholds_for_action_potential_spikes,
                                                   float *resting_potentials,
                                                   float* last_spike_time_of_each_neuron,
                                                   unsigned char* bitarray_of_neuron_spikes,
                                                   int bitarray_length,
                                                   int bitarray_maximum_axonal_delay_in_timesteps,
                                                   float current_time_in_seconds,
                                                   float timestep,
                                                   size_t total_number_of_neurons) {
      // Get thread IDs
      int idx = threadIdx.x + blockIdx.x * blockDim.x;
      while (idx < total_number_of_neurons) {
        if (membrane_potentials_v[idx] >= thresholds_for_action_potential_spikes[idx]) {

          // Set current time as last spike time of neuron
          last_spike_time_of_each_neuron[idx] = current_time_in_seconds;

          // Reset membrane potential
          membrane_potentials_v[idx] = resting_potentials[idx];

          // Get start of the given neuron's bits
          int neuron_id_spike_store_start = idx * bitarray_length;
          // Get offset depending upon the current timestep
          int offset_index = (int)(round((float)(current_time_in_seconds / timestep))) % bitarray_maximum_axonal_delay_in_timesteps;
          int offset_byte = offset_index / 8;
          int offset_bit_pos = offset_index - (8 * offset_byte);
          // Get the specific position at which we should be putting the current value
          unsigned char byte = bitarray_of_neuron_spikes[neuron_id_spike_store_start + offset_byte];
          // Set the specific bit in the byte to on 
          byte |= (1 << offset_bit_pos);
          // Assign the byte
          bitarray_of_neuron_spikes[neuron_id_spike_store_start + offset_byte] = byte;

        } else {
          // Get start of the given neuron's bits
          int neuron_id_spike_store_start = idx * bitarray_length;
          // Get offset depending upon the current timestep
          int offset_index = (int)(round((float)(current_time_in_seconds / timestep))) % bitarray_maximum_axonal_delay_in_timesteps;
          int offset_byte = offset_index / 8;
          int offset_bit_pos = offset_index - (8 * offset_byte);
          // Get the specific position at which we should be putting the current value
          unsigned char byte = bitarray_of_neuron_spikes[neuron_id_spike_store_start + offset_byte];
          // Set the specific bit in the byte to on 
          byte &= ~(1 << offset_bit_pos);
          // Assign the byte
          bitarray_of_neuron_spikes[neuron_id_spike_store_start + offset_byte] = byte;
        }

        idx += blockDim.x * gridDim.x;
      }
      __syncthreads();
    }

  } // ::Backend::CUDA
} // ::Backend
